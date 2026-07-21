"""The world: several levels, and the portals that join them.

A :class:`Level` is one maze plus a name. A :class:`LevelPortal` joins two
levels through a :class:`~.portal.PortalEnd` on each side. A portal end is an
entity that lives on the map like any item (see :mod:`.portal`); it is what you
see and step onto, and it carries how it is used (walk-on vs command) and
whether you may leave through it. The world only has to hold the *join* -- the
pair of ends and which levels they connect.

Positions here are ``(x, y)`` -- the same convention as entity location slots
and the opposite of numpy's ``maze.blocks[y, x]`` indexing.

Game-side module: no rendering library may be imported here.
"""
import numpy as np

from .array_cache import ArraySumCache
from .blocktypes import BlockTypes
from .events import Observable
from .layers import FieldLayer


#: Resolution of the sight fields relative to maze cells. One number, shared by
#: Level (which allocates the fields) and Scene (which composites them), so the
#: two cannot disagree about how big a level's fields are.
SIGHT_SUPERSAMPLE = 4


class Level:
    """One maze, under a name, plus everything sized against that maze.

    A Level owns *every* array sized to its maze -- the line-of-sight and
    memory fields, the composited lighting, the composited ``sight`` field the
    renderer uploads, and (injected by the display backend) the ``visibility``
    shadow provider. Nothing maze-sized lives on the scene "for whichever level
    is current"; it all lives here, on the level it belongs to.

    That is what makes the level the unit of consistency across threads. A
    reader -- the draw thread compositing a frame while the input thread walks
    the player to another level -- captures one Level reference and derives
    every shape from it. ``level.line_of_sight``, ``level.norm_light`` and
    ``level.visibility`` are all sized to ``level.maze`` by construction, so no
    interleaving can hand back a field and a maze that disagree. The worst a
    torn read costs is a frame drawn from a stale-but-internally-consistent
    level; :meth:`update_sight` turns even that into a fully-blocked (memory
    only) frame, because the player is not on the level being drawn.

    ``memory`` is per level for the same reason it is useful: what you saw of
    a level is a fact about that level, and survives going elsewhere and
    coming back.
    """

    #: sight memory fades to this fraction of itself per second (equivalent to
    #: the historical 0.999-per-frame decay at 60 fps)
    MEMORY_DECAY_RATE = 0.999 ** 60

    def __init__(self, name, maze, supersample=SIGHT_SUPERSAMPLE):
        self.name = name
        self.maze = maze
        self.world = None
        self.supersample = supersample

        # let anything holding a maze find the level it belongs to; this is
        # the hop that lets an entity ask about *its own* level's sight
        maze.level = self

        ms = maze.shape
        self.field_shape = (ms[0] * supersample, ms[1] * supersample, 3)
        self.memory = np.zeros(self.field_shape, dtype='float32')
        self.line_of_sight = np.zeros(self.field_shape, dtype='float32')

        # Light components whose global location is on this level; each Light
        # adds and removes itself as its host moves (see Light._register).
        self.lights = []

        # Fired when a light on this level changes what it emits. The display
        # backend connects this to its repaint (via Scene.request_redraw): the
        # level owns the decision to recomposite, the scene owns the repaint,
        # and neither the light nor the level needs a scene reference to reach
        # the other. See add_light/_light_changed.
        self.lighting_changed = Observable()

        # Composited lighting for this level, and its cross-frame caches. All
        # sized to this maze, all reached only through this level, so the
        # lights summed here (this level's) and the field they multiply (this
        # level's line of sight) can never be of two different shapes.
        self.norm_light = None
        self.light_cache = ArraySumCache()

        # Divisor mapping the composited log-lighting into 0-1. Held across
        # frames so a flickering light actually modulates the output:
        # renormalising per frame would divide the flicker straight back out
        # (the peak would pin to 1.0 no matter how the flame behaved). Dropped
        # on a genuine lighting change (entering the level, the viewer moving).
        self._light_norm = None

        # The composited sight field the renderer uploads. Owned by the level
        # so its identity is stable for a backend that captured it, and so it
        # is always the right shape for this maze.
        self.sight = FieldLayer('sight', shape=self.field_shape)

        # Shadow-map provider sized to this maze, injected by the display
        # backend when it builds this level's GL resources (see the vispy
        # renderer's _rebuild_for_level). Duck-typed render(pos, read=True) ->
        # (h, w, >=3) array; no rendering library is imported here to hold it.
        self.visibility = None

        # recompute line of sight on the next update (the viewer just arrived
        # or moved); set true so the first frame casts sight from scratch
        self._need_los_update = True

    def clear_line_of_sight(self):
        """Nothing on this level is in sight; the viewer has gone elsewhere.

        Written in place, so the array a concurrent reader holds stays the
        right shape throughout. Called when the level stops being displayed:
        with no player here, no torch on this level is being watched, which is
        what stops the flicker thread burning flames nobody can see.
        """
        self.line_of_sight[:] = 0

    def add_light(self, light):
        """Register *light* as shining on this level.

        Called by :meth:`Light._register` when a light's host moves onto this
        level, or when a map light is pinned here. Besides holding the light in
        ``lights`` for compositing, the level subscribes to the light's
        ``changed`` signal so that a change in the light's colour or brightness
        becomes stale lighting and a repaint here -- which is what lets a light
        announce it changed without holding any reference to the scene.
        """
        self.lights.append(light)
        light.changed.connect(self._light_changed)

    def remove_light(self, light):
        """Take *light* off this level; its host has moved elsewhere."""
        light.changed.disconnect(self._light_changed)
        self.lights.remove(light)

    def _light_changed(self):
        """A light on this level changed what it emits: recomposite and repaint.

        Runs on whichever thread set the light -- notably the torch flicker
        thread -- so it only nulls a reference (``invalidate_lighting``) and
        fires an observable, both safe off the main thread, exactly as the old
        direct calls to invalidate_lighting()/request_redraw() were.
        """
        self.invalidate_lighting()
        self.lighting_changed()

    def invalidate_lighting(self):
        """Discard the composited lighting; it is rebuilt on the next update.

        Called by a light on this level whose emitted light changed (colour,
        brightness). The kept ``_light_norm`` is deliberately *not* dropped, so
        a flickering flame modulates the output instead of renormalising away.
        """
        self.norm_light = None

    def invalidate_sight(self):
        """The viewer moved on this level: recast sight and rescale lighting.

        Drops the line of sight, the composited lighting, and the normalisation
        divisor, so the next update rebuilds all three for the new viewpoint.
        """
        self._need_los_update = True
        self.norm_light = None
        self._light_norm = None

    def enter(self):
        """Prepare this level to be shown, dropping every cross-frame cache.

        Called when the level becomes the displayed one. Blanks the composited
        field and the lighting caches so the first frame is built from scratch,
        matching what a freshly-entered level should look like.
        """
        self.norm_light = None
        self._light_norm = None
        self.light_cache = ArraySumCache()
        self._need_los_update = True
        self.sight.set_data(np.zeros(self.field_shape, dtype='float32'))

    def update_sight(self, dt, player):
        """Advance this level's sight/memory field by *dt* seconds, writing the
        result into ``self.sight``.

        Called once per rendered frame by the display backend, for the level it
        is currently showing. Everything read here -- the line-of-sight and
        lighting fields, the lights that feed them, the shadow provider -- is
        this one level's, sized to this one maze, so no interleaving with a
        level switch on another thread can compose arrays of two shapes.

        When *player* is not standing on this level the view is fully blocked:
        line of sight is all zero, so the composite collapses to the remembered
        field and no lights or shadow maps are consulted. That is what the
        renderer shows in the brief window after it has switched to a new level
        but before the player has been moved onto it -- the level's memory, for
        free, with no special case beyond the multiply by zero.
        """
        watched = player is not None and player.level is self

        if watched:
            if self._need_los_update:
                self.line_of_sight = player.line_of_sight().astype('float32', copy=False)
                self._need_los_update = False
            line_of_sight = self.line_of_sight

            # calculate lighting. Only this level's lights are consulted, so
            # their maps are all sized to this level -- no light on another
            # level can contribute a differently-shaped array to the sum.
            if self.norm_light is None:
                lights = []
                for light in self.lights:
                    light_map = light.lightmap(supersample=self.supersample)
                    if light_map is None:
                        continue
                    lights.append(light_map)
                if lights:
                    # ArraySumCache.sum_arrays asserts on an empty list, and a
                    # level may legitimately hold no light at all
                    summed = self.light_cache.sum_arrays(lights)
                    log_light = np.log(np.clip(summed * 10, 1, np.inf))
                    if self._light_norm is None:
                        self._light_norm = log_light.max()
                    self.norm_light = log_light / self._light_norm
                else:
                    self.norm_light = np.zeros(self.field_shape, dtype='float32')

            sight = line_of_sight * self.norm_light
        else:
            # fully blocked: the composite below reduces to memory alone
            line_of_sight = 0.0
            sight = 0.0

        # current sight is combination of lighting and LOS over memory
        self.sight.set_data(self.memory * (1 - line_of_sight) + sight)

        # forget
        self.memory *= self.MEMORY_DECAY_RATE ** dt

        # add sight to memory. Reducing the length-3 trailing axis with
        # sight.max(axis=2) is ~20x slower than maximum() applied pairwise.
        if watched:
            brightest = np.maximum(np.maximum(sight[:, :, 0], sight[:, :, 1]), sight[:, :, 2])
            self.memory[:, :, 2] = np.maximum(self.memory[:, :, 2], brightest)

    def __repr__(self):
        return "<Level %r %dx%d>" % ((self.name,) + self.maze.shape)


class LevelPortal:
    """A join between two levels, with a :class:`~.portal.PortalEnd` per side.

    The portal only ties the two ends together and lets you step from one to
    the other. Each end is an entity already standing on its own maze (see
    :mod:`.portal`); constructing the portal just records the pairing, so the
    end and the join agree on which two mouths are connected.
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
        """Record *portal* between its two already-placed ends."""
        self.portals.append(portal)
        return portal

    def link(self, end_a, end_b):
        """Join two :class:`~.portal.PortalEnd` entities into a portal.

        The ends are already standing on their mazes; this ties them together
        and registers the join. Returns the :class:`LevelPortal`.
        """
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
        """Return the :class:`~.portal.PortalEnd` at *pos* on *level*, or None.

        *level* may be a name, a Level, or a Maze -- the caller usually has
        whichever of those is nearest to hand. The dungeon master finds the end
        under the player through the maze inventory instead; this is for callers
        that have a level and a cell but not the maze's occupants to hand.
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
