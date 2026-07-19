"""The world: several levels, and the portals that join them.

A :class:`Level` is one maze plus a name. A :class:`LevelPortal` joins two
levels, and holds a :class:`PortalEnd` for each side describing *where* the
join lands on that level, *what block* stands there (a hole, stairs, a door),
and whether you may arrive through it. Stamping a portal writes each end's
block type into its level's maze, so the thing you see underfoot and the thing
that moves you between levels are one fact in one place.

How a portal is used is derived from the end's block type rather than stored
separately -- see :attr:`PortalEnd.command`. Stairs need the ``<`` / ``>``
command; a hole or a door acts the moment you step onto it.

Positions here are ``(x, y)`` -- the same convention as entity location slots
and the opposite of numpy's ``maze.blocks[y, x]`` indexing.

Game-side module: no rendering library may be imported here.
"""
from .blocktypes import BlockTypes


#: Block types that act as soon as you walk onto them; the rest need a command.
_WALK_ON_BLOCKS = ('hole', 'door')

#: Block type -> the command that operates it, for the ones that need one.
_BLOCK_COMMANDS = {'stairs_down': '>', 'stairs_up': '<'}


class Level:
    """One maze, under a name, belonging to a world."""

    def __init__(self, name, maze):
        self.name = name
        self.maze = maze
        self.world = None

    def __repr__(self):
        return "<Level %r %dx%d>" % ((self.name,) + self.maze.shape)


class PortalEnd:
    """One side of a :class:`LevelPortal`.

    *level* is the Level this end sits on, *pos* its ``(x, y)`` cell, and
    *blocktype* the name of the block stamped there ('hole', 'stairs_down',
    'stairs_up', 'door').

    *enterable* says whether the portal may be traversed **from** this side.
    Arriving is always allowed, so this is how a one-way drop is expressed: the
    hole is enterable where you fall in and not enterable where you land, which
    leaves the sewer's ceiling opening as something you can see and stand under
    but not climb back through.
    """

    def __init__(self, level, pos, blocktype, enterable=True):
        self.level = level
        self.pos = tuple(pos)
        self.blocktype = blocktype
        self.enterable = enterable
        self.portal = None

    @property
    def command(self):
        """The command that operates this end, or None if walking on it does.

        Derived from the block type so that changing what stands at an end
        changes how it is used, with nothing else to keep in step.
        """
        return _BLOCK_COMMANDS.get(self.blocktype)

    @property
    def walk_on(self):
        """True if stepping onto this end traverses the portal."""
        return self.blocktype in _WALK_ON_BLOCKS

    def __repr__(self):
        return "<PortalEnd %s@%s %s%s>" % (
            self.level.name, self.pos, self.blocktype,
            '' if self.enterable else ' (no entry)')


class LevelPortal:
    """A join between two levels, with one :class:`PortalEnd` per side.

    Constructing a portal stamps each end's block type into its level's maze,
    so the portal is the only thing that decides what stands at either mouth.
    """

    def __init__(self, end_a, end_b):
        self.ends = (end_a, end_b)
        for end in self.ends:
            end.portal = self

    def other(self, end):
        """Return the end opposite *end*."""
        a, b = self.ends
        if end is a:
            return b
        if end is b:
            return a
        raise ValueError("%r is not an end of %r" % (end, self))

    def stamp(self, blocktypes):
        """Write each end's block type into its level's maze."""
        for end in self.ends:
            x, y = end.pos
            end.level.maze.blocks[y, x] = blocktypes.id_of(end.blocktype)
            end.level.maze.invalidate_appearance()

    def __repr__(self):
        return "<LevelPortal %r <-> %r>" % self.ends


class World:
    """Every level, the portals between them, and which level is current.

    Owns the one shared :class:`~.blocktypes.BlockTypes` table: every maze in
    the world indexes the same table, so block ids mean the same thing on every
    level and the scene's glyph registry sees each block character once.
    """

    def __init__(self, blocktypes=None):
        self.blocktypes = blocktypes if blocktypes is not None else BlockTypes()
        self.levels = {}
        self.portals = []
        self.current = None

    def add_level(self, level):
        """Add *level* to the world; the first one added becomes current."""
        assert level.name not in self.levels, "duplicate level %r" % level.name
        self.levels[level.name] = level
        level.world = self
        if self.current is None:
            self.current = level
        return level

    def add_portal(self, portal):
        """Add *portal* and stamp its ends into their mazes."""
        self.portals.append(portal)
        portal.stamp(self.blocktypes)
        return portal

    def link(self, level_a, pos_a, blocktype_a, level_b, pos_b, blocktype_b,
             enterable_a=True, enterable_b=True):
        """Build, stamp and add a portal between two levels.

        Levels may be passed by name or as :class:`Level` objects.
        """
        end_a = PortalEnd(self.level(level_a), pos_a, blocktype_a, enterable_a)
        end_b = PortalEnd(self.level(level_b), pos_b, blocktype_b, enterable_b)
        return self.add_portal(LevelPortal(end_a, end_b))

    def level(self, level):
        """Resolve *level*, given as a name or a Level, to a Level."""
        return self.levels[level] if isinstance(level, str) else level

    def level_for_maze(self, maze):
        """Return the Level whose maze is *maze*, or None."""
        for level in self.levels.values():
            if level.maze is maze:
                return level
        return None

    def portal_end_at(self, level, pos):
        """Return the PortalEnd at *pos* on *level*, or None.

        *level* may be a name, a Level, or a Maze -- the caller usually has
        whichever of those is nearest to hand.
        """
        if not isinstance(level, (str, Level)):
            level = self.level_for_maze(level)
        else:
            level = self.level(level)
        pos = tuple(pos)
        for portal in self.portals:
            for end in portal.ends:
                if end.level is level and end.pos == pos:
                    return end
        return None
