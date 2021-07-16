# coding: utf8
import numpy as np
from .inventory import Inventory
from .location import Location
from .entity_type import EntityType
from .sprite import SingleCharSprite


class Item(object):

    name = "nondescript item"
    char = '?'
    mass = 0.0     # in kg
    length = 10.0  # in cm
    readable = False
    takable = False
    fg_color = (0, 0, 0.8, 1)
    bg_color = None
    light_source = False
    light_color = (10, 10, 10)

    def __init__(self, location, scene):
        self.scene = scene

        self.type = EntityType('item.' + self.name)
        self.inventory = Inventory(self, allowed_slots=[])
        self.location = Location(self, None, None)
        self.sprite = SingleCharSprite(self, zval=-0.1, char=self.char, fg_color=self.fg_color)

        scene.add_item(self)

        # for handling light sources
        self._shadow_map = None
        self._unscaled_light_map = None
        self._light_map = None

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
        self.sprite.position = (float('nan'),) * 3
        loc = self._location
        if isinstance(loc, (tuple, list, np.ndarray)):
            self.scene.items[self._location].remove(self)
        elif isinstance(loc, Player):
            loc.lose_item(self)
        self.location = None

    def shadow_map(self):
        if self._shadow_map is None:
            smap = self.scene.shadow_renderer.render(self.location.slot, read=True)[..., :3]
            self.set_shadow_map(smap)
            assert self._shadow_map is not None
        return self._shadow_map

    def set_shadow_map(self, smap):
        self._shadow_map = smap
        self._unscaled_light_map = None
        self._light_map = None

    def lightmap(self, supersample=1):
        if self._unscaled_light_map is None:
            ml = self.location.global_location
            if ml is None:
                return None
            
            (x,y) = ml.slot

            maze_shape = self.scene.maze.shape
            maze_pos = np.mgrid[0:maze_shape[0]*supersample, 0:maze_shape[1]*supersample].transpose(1, 2, 0)
            light_pos = np.array([[[y * supersample, x * supersample]]]) + (0.5 * supersample)
            dist2 = ((maze_pos - light_pos) ** 2).sum(axis=2) + 0.5  # 0.5 enforces height

            self._unscaled_light_map = self.shadow_map() / dist2[:, :, None]

        if self._light_map is None:
            self._light_map = self._unscaled_light_map * np.array(self.light_color)[None, None, :]

        return self._light_map

    def __repr__(self):
        return f"<{type(self).__name__} {self.name} {id(self)}>"


class Scroll(Item):

    name = "scroll of infinite recursion"
    char = u'次'
    readable = True
    takeable = True
    light_source = False
    mass = 0.05
    length = 20.0
    fg_color = (0.8, 0.8, 0.8, 1.0)

    def read(self, reader):
        self.scene.write("Unfortunately, you never learned to read. The scroll nevertheless appreciates your effort and self-destructs out of pity.") 
        self.destroy()


class Torch(Item):

    name = "torch"
    char = 't'
    readable = False
    takeable = True
    light_source = True
    mass = 0.5
    length = 30.0
    light_color = (10.0, 8.0, 2.0)
    fg_color = (1.0, 0.8, 0.2, 1.0)




from .player import Player



