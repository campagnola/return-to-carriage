# coding: utf8
import numpy as np


class Item(object):

    name = "nondescript item"
    char = '?'
    mass = 0.0   # in kg
    readable = False
    takable = False

    def __init__(self, location, scene):
        self._location = (None, None)
        self.scene = scene
        
        self.sprite = scene.txt.add_sprites((1,))        
        self.sprite.fgcolor = (0, 0, 0.6, 1)
        self.sprite.bgcolor = (0, 0, 0, 1)
        self.sprite.sprite = scene.atlas[self.char]
        
        self.location = location

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



class Scroll(Item):

    name = "scroll of infinite recursion"
    char = u'次'
    readable = True
    takable = True
    mass = 0.05

    def read(self, reader):
        self.scene.write("Unfortunately, you never learned to read. The scroll nevertheless appreciates your effort and self-destructs out of pity.") 
        self.destroy()
        

from .player import Player
