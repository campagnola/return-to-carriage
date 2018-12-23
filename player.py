# coding: utf8
import numpy as np


class Player(object):
    def __init__(self, scene):
        self.scene = scene
        self.zval = -0.2
        self.sprite = self.scene.txt.add_sprites((1,))
        self.sprite.sprite = self.scene.atlas.add_chars('&')
        self.sprite.fgcolor = (0, 0, 0.3, 1)
        self.sprite.bgcolor = (0.5, 0.5, 0.5, 1)
        self.position = (7, 7)
        
        self.inventory = []

    @property
    def position(self):
        return np.array(self._pos)
    
    @position.setter
    def position(self, p):
        self._pos = p
        self.sprite.position = tuple(p) + (self.zval,)

    def take(self, item):
        self.inventory.append(item)
        item.location = self

    def drop(self, item):
        self.inventory.remove(item)
        item.location = self.position
