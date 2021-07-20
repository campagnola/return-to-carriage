

class Entity:
    """A physical object in the game such as a player, item, mob, or maze."""
    def __init__(self, entity_type, obj_name=None):
        self.obj_name = obj_name
        self.type = EntityType.create(entity_type)

    def __repr__(self):
        return f"<{self.__class__.__name__} name={repr(self.obj_name)} type={self.type} id=0x{id(self):x}>"

    @property
    def components(self):
        return {k:v for k,v in self.__dict__.items() if isinstance(v, Component)}


class Component(Entity):
    """A sub-part of an entity"""
    def __init__(self, parent_entity, component_type, obj_name=None):
        Entity.__init__(self, entity_type="component."+component_type, obj_name=obj_name)
        self.parent_entity = parent_entity


class EntityType:
    """Used for describing an entity in the game.

    Examples::

        player.type = EntityType('player')
        item.type = EntityType('item.armor.soccer shin guard')
        mob.type = EntityType('mob.goblin.cave stinker')
    """
    @classmethod
    def create(cls, arg):
        if isinstance(arg, str):
            return EntityType(arg)
        elif isinstance(arg, EntityType):
            return arg
        else:
            raise TypeError("Invalid EntityType: %r" % arg)

    def __init__(self, type):
        assert isinstance(type, str)
        self.type = type.split('.')

    def isa(self, type):
        """Return whether this type is a (sub)type of another.

        EntityType('item.torch.burning_stick').isa('item.torch') => True
        """
        parts = type.split('.')
        return self.type[:len(parts)] == parts

    def __str__(self):
        return '.'.join(self.type)