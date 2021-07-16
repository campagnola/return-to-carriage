

class EntityType:
    """Used for describing an entity in the game.

    Examples::

        player.type = EntityType('player')
        item.type = EntityType('item.armor.soccer shin guard')
        mob.type = EntityType('mob.goblin.cave stinker')
    """
    def __init__(self, type):
        self.type = type.split('.')

    def isa(self, type):
        """Return whether this type is a (sub)type of another.

        EntityType('item.torch.burning_stick').isa('item.torch') => True
        """
        parts = type.split('.')
        return self.type[:len(parts)] == parts
