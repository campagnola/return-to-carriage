# coding: utf8
import vispy.scene, vispy.app
import numpy as np


class Monster(object):
    def __init__(self, position, scene):
        self._position = None
        self.scene = scene
        
        self.sprite = scene.txt.add_sprites((1,))        
        self.sprite.fgcolor = (0.6, 0.6, 0.6, 1)
        self.sprite.bgcolor = (0, 0, 0, 1)
        self.sprite.sprite = scene.atlas[u'Y']
        
        self.position = position
    
    def turn(self):
        l = list(self.position)
        l[1] -= 1
        self.position = l

    @property
    def position(self):
        return self._position
    
    @position.setter
    def position(self, pos):
        old_pos = self._position
        self._position = pos
        self.sprite.position = (pos[0], pos[1], -0.1)
        self.scene.monster_moved(self, old_pos)
