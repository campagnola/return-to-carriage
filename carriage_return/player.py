from .adaptation import EyeAdaptation
from .entity import Entity
from .errors import ActionError
from .inventory import Inventory
from .light import PointLight
from .location import Location
from .sprite import SingleCharSprite
from .units import lm
from .random import RandomDist


#: The 'glo' spell's light: bright, pure white. Equal RGB channels so it reads
#: white (a torch is warm ~(15, 12, 3) lm); a few times a torch's output lights
#: the room around the player without the fierce glare of a fireball (150 lm).
GLO_LIGHT_COLOR = (50 * lm, 50 * lm, 50 * lm)


class Player(Entity):
    def __init__(self, scene, obj_name=None):
        Entity.__init__(self, entity_type='player', obj_name=obj_name)
        self.scene = scene

        self.inventory = Inventory(self, slot_type=str, max_weight=40, max_length=100, allowed_slots=['right hand', 'left hand'])
        self.location = Location(self, None, None)
        # zval more negative than any other entity (monsters/items sit at -0.1)
        # so the player always draws on top when co-located.
        self.sprite = SingleCharSprite(self, zval=-0.2, char='&', fg_emission=(1, 1, 1), layer='actors')

        # Eye adaptation is player state, not level state: it persists across
        # levels so it rides the player through the hole -- eyes stay daylight-
        # adapted for the first moment in the dark sewer. It starts with no
        # reference and establishes one from the first scene it is shown; since
        # the player begins in home daylight, that first sight is what "adapted
        # to outdoor light" means (see adaptation.EyeAdaptation).
        self.adaptation = EyeAdaptation()

        # The 'glo' spell's light, or None while it is unlit. It is a point
        # light hosted on the player, so -- like eye adaptation -- it is player
        # state that follows the player everywhere and re-registers across
        # levels; it needs no inventory slot. Toggled by toggle_glo.
        self.glo = None

        self.base_perception = RandomDist('lognorm', mean=1, stdev=0.1)

        scene.player = self

    @property
    def perception(self):
        return self.base_perception()

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

    def toggle_glo(self):
        """Toggle the 'glo' spell: a bright white light carried by the player.

        Cast once, a white point light hangs off the player and lights the room
        wherever they go; cast again, it goes out. The light follows the player
        the same way a carried torch does -- it hosts on the player's location,
        so it needs no inventory slot and rides through the hole to the next
        level. Returns True if it is now lit, False if it was just snuffed.

        Building or tearing down a light does not on its own mark the level's
        lighting stale, so each branch nudges the level to recomposite and
        repaint (mirroring Spell._relight / Spell.destroy).
        """
        if self.glo is not None:
            level = self.glo.level
            self.glo.destroy()
            self.glo = None
            if level is not None:
                level.invalidate_lighting()
                level.lighting_changed()
            return False
        self.glo = PointLight(self, color=GLO_LIGHT_COLOR)
        self.glo.changed()
        return True

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
        # in a pocket has no shadow map to share. The glo light hangs off the
        # player too, sitting on the player's own cell, so it shares the very
        # same shadow map.
        lights = [getattr(item, 'light', None)
                  for item in self.inventory.all_entities()]
        lights.append(self.glo)
        for light in lights:
            if isinstance(light, PointLight):
                light.set_shadow_map(smap)
        return smap / 255.0
