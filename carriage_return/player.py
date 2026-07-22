from .adaptation import EyeAdaptation
from .entity import Entity
from .errors import ActionError
from .inventory import Inventory
from .light import PointLight
from .location import Location
from .sprite import SingleCharSprite


class Player(Entity):
    def __init__(self, scene, obj_name=None):
        Entity.__init__(self, entity_type='player', obj_name=obj_name)
        self.scene = scene

        self.inventory = Inventory(self, slot_type=str, max_weight=40, max_length=100, allowed_slots=['right hand', 'left hand'])
        self.location = Location(self, None, None)
        # zval more negative than any other entity (monsters/items sit at -0.1)
        # so the player always draws on top when co-located.
        self.sprite = SingleCharSprite(self, zval=-0.2, char='&', layer='actors')

        # Eye adaptation is player state, not level state: it persists across
        # levels so it rides the player through the hole -- eyes stay daylight-
        # adapted for the first moment in the dark sewer. It starts with no
        # reference and establishes one from the first scene it is shown; since
        # the player begins in home daylight, that first sight is what "adapted
        # to outdoor light" means (see adaptation.EyeAdaptation).
        self.adaptation = EyeAdaptation()

        scene.player = self

    def take(self, item):
        """Move *item* from the maze into this player's inventory.

        Raises ActionError (with a user-facing ``reason``) when the item
        cannot be taken. Returns the inventory slot used.
        """
        if not item.takeable:
            raise ActionError("The %s stays resolutely where it is." % item.description)
        reasons = []
        for slot in self.inventory.allowed_slots:
            if len(self.inventory[slot]) > 0:
                continue
            allowed, reasons = self.inventory.check_entity_add(item, slot, actor=self)
            if allowed:
                item.location.update(self, slot)
                return slot
        if reasons:
            raise ActionError("You cannot take the %s: %s." % (item.description, '; '.join(reasons)))
        raise ActionError("Your hands are full.")

    def drop(self, item):
        """Move *item* from this player's inventory onto the ground below.

        Raises ActionError when the player is not holding the item.
        """
        if item.location.container is not self:
            raise ActionError("You are not holding the %s." % item.description)
        item.location.update(self.location.container, self.location.slot)

    def read(self, item):
        """Read *item*. Reading is an act with in-game constraints and
        consequences that live on the item; ``item.read`` shows any text and
        applies effects, and returns whatever it opens (e.g. a pager session).
        """
        return item.read(self)

    @property
    def level(self):
        """The Level this player is standing on, or None if nowhere.

        Read as one ``(maze, slot)`` so the maze and the position always belong
        together even while the player is being moved between levels.
        """
        ml = self.location.global_location
        if ml is None:
            return None
        maze = ml.container
        return None if maze is None else maze.level

    def line_of_sight(self):
        pos = self.location.global_location.slot
        smap = self.level.visibility.render(pos, read=True)[:, :, :3]
        # carried point lights share the player's shadow map; they expect it in
        # the same 0-255 scale that PointLight.shadow_map() renders for itself.
        # Only point sources cast shadows -- an ambient or array light carried
        # in a pocket has no shadow map to share.
        for item in self.inventory.all_entities():
            light = getattr(item, 'light', None)
            if isinstance(light, PointLight):
                light.set_shadow_map(smap)
        return smap / 255.0
