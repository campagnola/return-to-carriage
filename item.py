# coding: utf8
import numpy as np


class Item(object):
    def __init__(self, location, scene):
        self._location = None
        self.name = "a scroll of infinite recursion"
        self.scene = scene
        
        self.sprite = scene.txt.add_sprites((1,))        
        self.sprite.fgcolor = (0, 0, 0.6, 1)
        self.sprite.bgcolor = (0, 0, 0, 1)
        self.sprite.sprite = scene.atlas[u'次']
        
        self.location = location

    @property
    def location(self):
        return self._location
    
    @location.setter
    def location(self, loc):
        if self._location is not None:
            self.scene.items[self._location].remove(self)
        
        self._location = loc
        self.scene.items.setdefault(loc, [])
        self.scene.items[loc].append(self)
        if isinstance(loc, (tuple, list, np.ndarray)):
            self.sprite.position = (loc[0], loc[1], -0.1)
        elif isinstance(loc, Player):
            self.sprite.position = (float('nan'),) * 3
        else:
            raise TypeError('location must be x,y position or Player')
