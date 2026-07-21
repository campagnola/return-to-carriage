# coding: utf8
"""Portal ends: the mouths of a portal, as entities that live on the map.

A portal end is a physical thing you see and step onto, so it is an
:class:`~.entity.Entity` like an item or a mob -- not a row in a side table.
Being an entity, it gets the whole entity toolkit for free: a
:class:`~.location.Location` (which registers it in its maze's inventory, so the
dungeon master finds it under the player's feet), an optional
:class:`~.sprite.SingleCharSprite` for its glyph, optional lights, and
:attr:`~.entity.Entity.blocks_movement` for whether it stops a mover.

What *kind* of mouth it is -- a hole, a stair -- is a subclass carrying its own
defaults (its glyph, whether it acts on a step or waits for a command, the
message it refuses entry with), so placing a stairway does not mean respelling
``>`` every time. A one-off look is an instance: pass ``char=None`` to hide the
glyph, or attach extra components (a shaft of daylight, say) after construction.
The two ends of a join are tied together by :class:`~.world.LevelPortal`.

Game-side module: no rendering library is imported here.
"""
from .entity import Entity
from .inventory import Inventory
from .location import Location
from .sprite import SingleCharSprite


#: Sentinel for "use the class's default glyph"; distinct from ``char=None``,
#: which is an explicit request for no glyph at all (an invisible end).
_INHERIT = object()


class PortalEnd(Entity):
    """One mouth of a portal, standing on a maze cell.

    Subclass to define a kind of mouth (see :class:`Hole`, :class:`StairsDown`,
    :class:`StairsUp`, :class:`Door`); the class attributes below are its
    defaults. Construct with a *location* ``(maze, (x, y))`` and the *scene*,
    exactly like an :class:`~.item.Item`. *enterable* says whether the portal
    may be traversed **from** this side -- arriving is always allowed, so a
    one-way drop is enterable where you fall in and not where you land. Pass
    *char* to override the kind's glyph, including ``None`` for an invisible end
    whose floor shows through and whose only mark is the light it carries.
    """

    #: glyph drawn over the floor; None => nothing drawn, the floor shows through
    char = None
    #: colour of the glyph; the background is left transparent so the lit floor
    #: shows through behind it
    fg_color = (0.7, 0.7, 0.6, 1.0)
    #: True => stepping onto the end traverses immediately (a hole, a door)
    walk_on = False
    #: the command that operates the end ('<' / '>'), or None if a step does
    command = None
    #: shown when you try to enter from a side you cannot
    refusal = "You cannot go that way."
    #: short name / description
    name = "portal"

    def __init__(self, location, scene, enterable=True, char=_INHERIT, obj_name=None):
        Entity.__init__(self, entity_type='portal.' + self.name, obj_name=obj_name)
        self.scene = scene
        self.enterable = enterable
        self.portal = None                       # set by LevelPortal

        # an entity like an item: a (empty) inventory and a location. Setting the
        # location registers this end in its maze's inventory, which is how the
        # dungeon master finds a portal end under the player's feet.
        self.inventory = Inventory(self, allowed_slots=[])
        self.location = Location(self, None, None)

        glyph = self.char if char is _INHERIT else char
        self.sprite = None
        if glyph is not None:
            # zval sits above the floor scenery but below items and mobs, so a
            # dropped torch or a monster draws on top of the stair it stands on.
            self.sprite = SingleCharSprite(self, zval=-0.05, char=glyph,
                                           fg_color=self.fg_color, layer='items')

        # lights this end carries (a daylight shaft, a glow). Built by the
        # caller and appended here so they are held for the end's lifetime; each
        # is hosted by this entity, so it tracks this cell and this level.
        self.lights = []

        if location is not None:
            self.location.update(*location)

    def on_walked_on(self, mover, dm):
        """A walk-on end (a hole, a door) asks to send the mover through.

        The dungeon master decides whether the traversal actually happens; this
        only requests it. A command-operated end (stairs) does nothing here --
        it waits for its command.
        """
        if self.walk_on:
            dm.request_traverse(mover, self)

    def on_command(self, actor, command, dm):
        """A command-operated end asks to traverse when its command is given.

        Returns True when the command is this end's -- the end has responded,
        whatever the dungeon master then decides -- so the caller stops looking.
        """
        if self.command is not None and command == self.command:
            dm.request_traverse(actor, self)
            return True
        return False

    @property
    def level(self):
        """The Level this end stands on, or None."""
        container = self.location.container
        return None if container is None else container.level

    @property
    def pos(self):
        """This end's ``(x, y)`` cell, or None."""
        return self.location.slot

    @property
    def description(self):
        return self.name

    def __repr__(self):
        lvl = self.level.name if self.level is not None else None
        return "<%s %s@%s%s>" % (self.__class__.__name__, lvl, self.pos,
                                 '' if self.enterable else ' (no entry)')


class Hole(PortalEnd):
    """A hole you drop through. Acts the moment you step on it."""
    name = "hole"
    char = 'O'
    fg_color = (0.4, 0.4, 0.4, 1.0)
    walk_on = True
    refusal = "The opening is far above you; there is no way back up."


class StairsDown(PortalEnd):
    """Stairs leading down; operated with the ``>`` command."""
    name = "stairs down"
    char = '>'
    fg_color = (0.7, 0.7, 0.6, 1.0)
    command = '>'


class StairsUp(PortalEnd):
    """Stairs leading up; operated with the ``<`` command."""
    name = "stairs up"
    char = '<'
    fg_color = (0.7, 0.7, 0.6, 1.0)
    command = '<'


class Door(PortalEnd):
    """A doorway you pass through by stepping onto it."""
    name = "door"
    char = '+'
    fg_color = (0.6, 0.4, 0.15, 1.0)
    walk_on = True
