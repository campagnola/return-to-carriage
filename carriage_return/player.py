# coding: utf8
from .entity import Entity
from .inventory import Inventory
from .location import Location
from .sprite import SingleCharSprite


class Player(Entity):
    def __init__(self, scene, obj_name=None):
        Entity.__init__(self, entity_type='player', obj_name=obj_name)
        self.scene = scene

        self.inventory = Inventory(self, slot_type=str, max_weight=40, max_length=100, allowed_slots=['right hand', 'left hand'])
        self.location = Location(self, None, None)
        self.sprite = SingleCharSprite(self, zval=-0.1, char='&')

        scene.player = self

    def read_item(self, item=None):
        if item is None:
            readables = [i for i in self.inventory if i.readable]
            self.scene.user_request_item("What item do you want to read?", readables, self.read_item)
            return
        
        item.read(self)

    def line_of_sight(self):
        pos = self.location.global_location.slot
        los = self.scene.shadow_renderer.render(pos, read=True)[:, :, :3] / 255.0
        for item in self.inventory.all_entities():
            if isinstance(item, Item) and item.light_source:
                item.set_shadow_map(los)
        return los


from .item import Item
