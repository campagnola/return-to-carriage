# coding: utf8
import numpy as np


class Item(object):

    name = "nondescript item"
    char = '?'
    mass = 0   # in kg
    readable = False
    takable = False

    def __init__(self, location, scene):
        self._location = None
        self.scene = scene
        
        self.sprite = scene.txt.add_sprites((1,))        
        self.sprite.fgcolor = (0, 0, 0.6, 1)
        self.sprite.bgcolor = (0, 0, 0, 1)
        self.sprite.sprite = scene.atlas[self.char]
        
        self.location = location

    @property
    def location(self):
        return self._location
    
    @location.setter
    def location(self, loc):
        if isinstance(self._location, (tuple, list, np.ndarray)):
            self.scene.items[self._location].remove(self)
        
        self._location = loc
        if isinstance(loc, (tuple, list, np.ndarray)):
            self.scene.items.setdefault(loc, [])
            self.scene.items[loc].append(self)
            self.sprite.position = (loc[0], loc[1], -0.1)
        elif isinstance(loc, Player):
            self.sprite.position = (float('nan'),) * 3
        else:
            raise TypeError('location must be x,y position or Player')

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
        

from player import Player
