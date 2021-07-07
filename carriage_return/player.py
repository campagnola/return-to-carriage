# coding: utf8
import numpy as np
from collections import OrderedDict
from .inventory import Inventory
from .location import Location
from .entity_type import EntityType


class Player(object):
    def __init__(self, scene):
        self.scene = scene

        self.type = EntityType('player')
        self.inventory = Inventory(self, max_weight=40, max_length=100, allowed_locations=['right hand', 'left hand'])
        self.location = Location(None, None)

        self.zval = -0.2
        self.sprite = self.scene.txt.add_sprites((1,))
        self.sprite.sprite = self.scene.atlas.add_chars('&')
        self.sprite.fgcolor = (0, 0, 0.3, 1)
        self.sprite.bgcolor = (0.5, 0.5, 0.5, 1)
        

    @property
    def position(self):
        return np.array(self._pos)
    
    @position.setter
    def position(self, p):
        self._pos = p
        self.sprite.position = tuple(p) + (self.zval,)

    def take(self, item):
        for k in self.inventory:
            if self.inventory[k] is None:
                self.inventory[k] = item
                item.location = (self, k)
                self.scene.console.write("Taken: %s" % item.description)
                return
        self.scene.console.write("For lack of another letter in the alphabet, you decline to take this item.")

    def drop(self, item):
        self.inventory.pop(item.location[1])
        item.location = (self.scene, tuple(self.position))
        self.scene.console.write("Dropped: %s" % item.description)

    def read_item(self, item=None):
        if item is None:
            readables = [i for i in self.inventory if i.readable]
            self.scene.user_request_item("What item do you want to read?", readables, self.read_item)
            return
        
        item.read(self)

    def lose_item(self, item):
        self.inventory.remove(item)

    def line_of_sight(self):
        los = self.scene.shadow_renderer.render(self.position, read=True) / 255.0
        for item in self.inventory.all_items():
            if isinstance(item, Item) and item.light_source:
                item.set_shadow_map(los)
        return los


from .item import Item
