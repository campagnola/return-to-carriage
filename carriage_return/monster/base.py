# coding: utf8
"""The :class:`Monster` base: an entity that stands in the maze and takes turns."""
from ..entity import Entity
from ..location import Location
from ..sprite import SingleCharSprite
from ..stats import Stats
from .behaviour import Idle


class Monster(Entity):
    """A mob standing on a maze cell, acting once per player turn.

    The monster is a body; what it does is its ``behaviour``'s choice, and
    whether that happens is the dungeon master's (see
    :meth:`.dm.DungeonMaster.request_monster_move`). It takes turns only while
    it is on the player's level (see :meth:`.dm.DungeonMaster.end_turn`), so a
    level the player is not on stays exactly as it was left.

    Its sprite is an ordinary reflective glyph on the ``actors`` layer, lit by
    the level's lighting and gated by line of sight like everything else: a
    monster standing in the dark cannot be seen. A glowing monster sets
    ``fg_emission``.
    """

    name = "monster"
    char = 'Y'
    fg_color = (0.6, 0.6, 0.6, 1)
    bg_color = None
    #: Light the glyph emits on its own (RGB, linear), or None for a plain
    #: reflective monster that shows only where it is lit.
    fg_emission = None

    #: a monster occupies its cell: neither the player nor another monster may
    #: step onto it
    blocks_movement = True

    def __init__(self, location, scene, behaviour=None, obj_name=None):
        Entity.__init__(self, entity_type='mob.monster.' + self.name, obj_name=obj_name)
        self.scene = scene
        self.behaviour = behaviour if behaviour is not None else Idle()
        self.stats = Stats(self)

        self.location = Location(self, None, None)
        # zval matches items (-0.1); the player (-0.2) draws on top of both
        self.sprite = SingleCharSprite(self, zval=-0.1, char=self.char,
                                       fg_color=self.fg_color, bg_color=self.bg_color,
                                       fg_emission=self.fg_emission, layer='actors')

        scene.add_monster(self)

        if location is not None:
            self.location.update(*location)

    @property
    def description(self):
        return self.name

    @property
    def level(self):
        """The Level this monster is standing on, or None if nowhere."""
        ml = self.location.global_location
        if ml is None or ml.container is None:
            return None
        return ml.container.level

    def take_turn(self, dm):
        """Act once: ask the behaviour for a step and request it from *dm*.

        Returns whether the monster moved.
        """
        step = self.behaviour.choose_step(self, dm)
        if step is None:
            return False
        x, y = self.location.slot
        return dm.request_monster_move(self, (x + step[0], y + step[1]))

    def destroy(self):
        """Remove this monster from the game."""
        self.scene.monsters.remove(self)
        self.location.update(None, None)
        self.sprite.hide()
