

class Inventory:
    """Any collection of entities with multiple storage slots

    - Player inventory
    - Items in maze
    - Monster inventory
    - Container contents
    """
    def __init__(self, entity, slot_type=None, max_weight=None, max_length=None, allowed_slots=None):
        self.entity = entity
        self.slot_type = slot_type or (lambda x: x)
        self.max_weight = max_weight
        self.max_length = max_length
        self.allowed_slots = allowed_slots
        self.slots = {}

    def check_entity_add(self, entity_to_add, slot, actor):
        """Return True if *entity* may be added to *slot*

        Also return a list of reasons.
        """
        slot = self.slot_type(slot)
        allowed = True
        reasons = []
        if not self.has_slot(slot):
            allowed = False
            reasons.append("slot %r is not allowed" % slot)
        if self.max_weight is not None and entity_to_add.weight + self.weight > self.max_weight:
            allowed = False
            reasons.append("entity is too heavy")
        if self.max_length is not None and entity_to_add.length > self.max_length:
            allowed = False
            reasons.append("entity is too large")
        
        return allowed, reasons

    def has_slot(self, slot):
        slot = self.slot_type(slot)
        if self.allowed_slots is not None and slot not in self.allowed_slots:
            return False
        return True

    def check_entity_remove(self, entity_to_remove, slot, actor):
        """Return True if *entity* may be removed from *slot*

        Also return a list of reasons.
        """
        slot = self.slot_type(slot)
        return (True, [])

    def _add_entity(self, entity, slot):
        # This should only be called indirectly via Location.update
        slot = self.slot_type(slot)
        self.slots.setdefault(slot, []).append(entity)

    def _remove_entity(self, entity, slot):
        # This should only be called indirectly via Location.update
        slot = self.slot_type(slot)
        self[slot].remove(entity)

    def __getitem__(self, slot):
        slot = self.slot_type(slot)
        assert self.has_slot(slot)
        return self.slots.get(slot, [])

    @property
    def weight(self):
        return sum([i.weight for i in self.all_entities()])

    def all_entities(self):
        for slot, entities in self.slots.items():
            for i in entities:
                yield i
