

class Entity:
    """A physical object in the game such as a player, item, mob, or maze."""

    #: Whether standing on this entity's cell is blocked. It is combined with
    #: the terrain and every other entity on the cell when deciding if a move is
    #: legal (see :meth:`.dm.DungeonMaster.walkable`), so blocking is a property
    #: any entity can carry, not a fact baked into the terrain grid. Most things
    #: you walk over -- items, an open portal mouth -- leave it False; a shut
    #: door or a boulder sets it True.
    blocks_movement = False

    def __init__(self, entity_type, obj_name=None):
        self.obj_name = obj_name
        self.type = EntityType.create(entity_type)

    def __repr__(self):
        return f"<{self.__class__.__name__} name={repr(self.obj_name)} type={self.type} id=0x{id(self):x}>"

    @property
    def components(self):
        return {k:v for k,v in self.__dict__.items() if isinstance(v, Component)}

    def has_component(self, name):
        return name in self.components

    # -- interaction ---------------------------------------------------------
    # How an entity responds to being acted upon. The dungeon master drives
    # these -- it does not know what any particular entity is or does; it asks.
    # An entity reacts by calling back to the dungeon master to request an
    # action (a hole asks to be traversed), and the dungeon master keeps final
    # say over whether that action happens. Both default to doing nothing, so
    # only entities that care override them.

    def on_walked_on(self, mover, dm):
        """React to *mover* stepping onto this entity's cell.

        Called for every entity on the cell a mover just entered. The entity
        decides what happens and may call back to *dm* to request an action.
        Default: nothing.
        """

    def on_command(self, actor, command, dm):
        """React to *actor* issuing *command* while on this entity's cell.

        Return True if this entity handled the command; the dungeon master asks
        each entity on the cell in turn and stops at the first that does.
        Default: not handled.
        """
        return False


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