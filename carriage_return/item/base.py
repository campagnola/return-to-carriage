# coding: utf8
"""The :class:`Item` base: an entity that lives on the map and may be taken."""
from ..entity import Entity
from ..inventory import Inventory
from ..location import Location
from ..sprite import SingleCharSprite


class Item(Entity):

    name = "nondescript item"
    char = '?'
    mass = 0.0     # in kg
    length = 10.0  # in cm
    readable = False
    takeable = False
    fg_color = (0, 0, 0.8, 1)
    bg_color = None
    #: Light the glyph emits on its own (RGB, linear), or None for a plain
    #: reflective item. An emitter (see Torch) glows independently of the light
    #: falling on it rather than merely reflecting its surroundings.
    emission = None

    def __init__(self, location, scene, obj_name=None):
        Entity.__init__(self, entity_type='item.' + self.name, obj_name=obj_name)
        self.scene = scene

        self.inventory = Inventory(self, allowed_slots=[])
        self.location = Location(self, None, None)
        self.sprite = SingleCharSprite(self, zval=-0.1, char=self.char,
                                       fg_color=self.fg_color, emission=self.emission,
                                       layer='items')

        # A plain item emits no light. An item that shines builds its own
        # Light in its __init__ (see Torch) and drives its colour/brightness;
        # the Light handles the rest of being a light source -- level tracking,
        # shadow maps, the emitted map.
        self.light = None

        scene.add_item(self)

        if location is not None:
            self.location.update(*location)

    @property
    def weight(self):
        # allows us to change gravity later..
        return self.mass

    @property
    def description(self):
        """A short description of this item
        """
        # todo: include minimal detail, custom naming, etc.
        return self.name

    def destroy(self):
        """Remove this item from the game.
        """
        self.scene.items.remove(self)
        if self.light is not None:
            self.light.destroy()
        self.location.update(None, None)
        self.sprite.hide()
