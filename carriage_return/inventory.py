

class Inventory:
    """Any collection of items with multiple storage locations

    - Player inventory
    - Items in maze
    - Monster inventory
    - Container contents
    """
    def __init__(self, parent, max_weight=None, max_length=None, allowed_locations=None):
        self.parent = parent
        self.max_weight = max_weight
        self.max_length = max_length
        self.allowed_locations = allowed_locations
        self.items = {}

    def add_item(self, item, location):
        if self.allowed_locations is not None:
            assert location in self.allowed_locations
        if self.max_weight is not None:
            assert item.weight + self.weight <= self.max_weight
        if self.max_length is not None:
            assert item.length <= self.max_length
        self.items.setdefault(location, []).append(item)
        item._location_changed((self.parent, location))

    def remove_item(self, item, location):
        self[location].remove(item)

    def __getitem__(self, location):
        if self.allowed_locations is not None:
            assert location in self.allowed_locations
        return self.items.get(location, [])

    @property
    def weight(self):
        return sum([i.weight for i in self.all_items()])

    def all_items(self):
        for slot, items in self.items.items():
            for i in items:
                yield i
