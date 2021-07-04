# coding: utf8
import numpy as np


class Item(object):

    name = "nondescript item"
    char = '?'
    mass = 0.0   # in kg
    readable = False
    takable = False
    fg_color = (0, 0, 0.8, 1)
    bg_color = None
    light_source = False
    light_color = (10, 10, 10)

    def __init__(self, location, scene):
        self._location = (None, None)
        self.scene = scene

        # for handling light sources
        self._shadow_map = None
        self._unscaled_light_map = None
        self._light_map = None

        self.sprite = scene.txt.add_sprites((1,))        
        self.sprite.fgcolor = self.fg_color
        self.sprite.sprite = scene.atlas[self.char]

        self.location = location

    def _maze_bg_color(self):
        return self.location[0].maze.bg_color[self.location[1]]

    @property
    def description(self):
        """A short description of this item
        """
        # todo: include minimal detail, custom naming, etc.
        return self.name

    @property
    def location(self):
        return self._location
    
    @location.setter
    def location(self, loc):
        """Set item location.

        *loc* should be a tuple (container, slot) where container is a Scene, Player, etc and slot
        is the (i, j) location within the scene or an inventory slot letter.
        """
        oldloc = self._location
        if oldloc[0] is self.scene:
            self.scene.items[oldloc[1]].remove(self)
        
        self._location = loc
        if loc[0] is self.scene:
            self.scene.items.setdefault(loc[1], [])
            self.scene.items[loc[1]].append(self)
            self.sprite.position = (loc[1][0], loc[1][1], -0.1)
        elif isinstance(loc[0], Player):
            self.sprite.position = (float('nan'),) * 3
        else:
            raise Exception("Not sure what to do with location: %s" % str(loc))

        self.sprite.bgcolor = self.bg_color or self._maze_bg_color()


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
            self._shadow_map = self.scene.shadow_renderer.render(self.location[1], read=True)[..., :3]
            assert self._shadow_map is not None
        return self._shadow_map

    def lightmap(self):
        if self._unscaled_light_map is None:
            scene = self.location[0]
            x,y = self.location[1]

            maze_shape = scene.maze.shape
            maze_pos = np.mgrid[0:maze_shape[0], 0:maze_shape[1]].transpose(1, 2, 0)
            dist2 = ((maze_pos - np.array([[[y, x]]])) ** 2).sum(axis=2) + 0.5  # 0.5 enforces height

            self._unscaled_light_map = self.shadow_map() / dist2[:, :, None]

        if self._light_map is None:

            self._light_map = self._unscaled_light_map * np.array(self.light_color)[None, None, :]

        return self._light_map




class Scroll(Item):

    name = "scroll of infinite recursion"
    char = u'次'
    readable = True
    takeable = True
    light_source = False
    mass = 0.05

    def read(self, reader):
        self.scene.write("Unfortunately, you never learned to read. The scroll nevertheless appreciates your effort and self-destructs out of pity.") 
        self.destroy()


class Torch(Item):

    name = "torch"
    char = 't'
    readable = False
    takeable = True
    light_source = True
    mass = 0.2
    fg_color = (1, 0.8, 0.2, 1)




from .player import Player



