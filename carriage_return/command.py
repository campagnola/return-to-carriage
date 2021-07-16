

class Command:
    """A request to change the state of the world.

    Commands are submitted to the DungeonMaster, which decides on the actual outcome and updates the 
    world state.

    Parameters
    ----------
    actor : entity
        The entity causing the change
    action : str
        The type of change requested
    operands : tuple | None
        The objects to act on (action-dependent)
    clauses : list | None
        Extra clauses that modify the action
    """
    def __init__(self, actor, action, operands=None, clauses=None):
        self.actor = actor
        self.action = action
        self.operands = operands
        self.clauses = clauses

    @classmethod
    def move(cls, actor, entity, location):
        """Return a command that requests *actor* to move *entity* to a new *location*.
        """
        return Command(actor, "move", (entity, location))
