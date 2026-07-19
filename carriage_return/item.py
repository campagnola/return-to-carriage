# coding: utf8
import threading
import time
import weakref

import numpy as np

from .entity import Entity
from .inventory import Inventory
from .location import Location
from .sprite import SingleCharSprite


class Item(Entity):

    name = "nondescript item"
    char = '?'
    mass = 0.0     # in kg
    length = 10.0  # in cm
    readable = False
    takeable = False
    fg_color = (0, 0, 0.8, 1)
    bg_color = None
    light_source = False
    light_color = (10, 10, 10)

    def __init__(self, location, scene, obj_name=None):
        Entity.__init__(self, entity_type='item.' + self.name, obj_name=obj_name)
        self.scene = scene

        self.inventory = Inventory(self, allowed_slots=[])
        self.location = Location(self, None, None)
        self.sprite = SingleCharSprite(self, zval=-0.1, char=self.char, fg_color=self.fg_color, layer='items')

        scene.add_item(self)

        # for handling light sources
        self._shadow_map = None
        self._unscaled_light_map = None
        self._light_map = None
        self._brightness = 1.0
        self.location.global_changed.connect(self._location_changed)

        if location is not None:
            self.location.update(*location)

    def _location_changed(self, event):
        self._shadow_map = None
        self._unscaled_light_map = None
        self._light_map = None

    @property
    def brightness(self):
        """Scale applied to ``light_color``; 1.0 is this item's nominal output.

        Setting it discards only the colour scaling of the cached light map --
        the shadow map and distance falloff underneath survive -- then tells
        the scene its lighting is stale and asks the display to repaint.
        """
        return self._brightness

    @brightness.setter
    def brightness(self, value):
        if value == self._brightness:
            return
        self._brightness = value
        self._light_map = None
        self.scene.invalidate_lighting()
        self.scene.request_redraw()

    @property
    def weight(self):
        # allows us to change gravity later..
        return self.mass

    @property
    def description(self):
        """A short description of this item
        """
        # todo: include minimal detail, custom naming, etc.
        return self.name

    def destroy(self):
        """Remove this item from the game.
        """
        self.scene.items.remove(self)
        self.location.update(None, None)
        self.sprite.hide()

    def shadow_map(self, slot):
        if self._shadow_map is None:
            smap = self.scene.visibility.render(slot, read=True)[..., :3]
            self.set_shadow_map(smap)
            assert self._shadow_map is not None
        return self._shadow_map

    def set_shadow_map(self, smap):
        self._shadow_map = smap
        self._unscaled_light_map = None
        self._light_map = None

    def in_player_sight(self):
        """True if the player can currently see the cell this item occupies.

        Sampled from the line-of-sight field the scene last composited, so it
        costs one array lookup and is safe to call from an animator thread
        (the field is replaced wholesale, never edited in place).

        An item on a level the scene is not displaying is never in sight -- and
        must return before the lookup, because the sight field is sized to the
        *current* level, so another level's coordinates may not even index it.
        """
        ml = self.location.global_location
        if ml is None or ml.container is not self.scene.maze:
            return False
        x, y = ml.slot
        ss = self.scene.supersample
        return self.scene.line_of_sight[y * ss, x * ss].max() > 0

    def lightmap(self, supersample=1):
        # A light on another level contributes nothing here, and its map would
        # be the wrong shape anyway: the field below is sized to scene.maze.
        ml = self.location.global_location
        if ml is None or ml.container is not self.scene.maze:
            return None

        if self._unscaled_light_map is None:
            (x,y) = ml.slot

            maze_shape = self.scene.maze.shape
            maze_pos = np.mgrid[0:maze_shape[0]*supersample, 0:maze_shape[1]*supersample].transpose(1, 2, 0)
            light_pos = np.array([[[y * supersample, x * supersample]]]) + (0.5 * supersample)
            dist2 = ((maze_pos - light_pos) ** 2).sum(axis=2) + 0.5  # 0.5 enforces height
            dist2 = dist2.astype('float32')

            # float32 throughout: these maps are rescaled and composited every
            # frame for flickering lights, and float64 doubles that cost
            self._unscaled_light_map = self.shadow_map(ml.slot) / dist2[:, :, None]

        # The shadow map and distance falloff above are cached until the item
        # moves; a brightness change only rescales that cached map. Held in a
        # local because an animator thread may null the cache at any moment --
        # the worst that costs is one frame at the previous brightness.
        light_map = self._light_map
        if light_map is None:
            color = np.array(self.light_color, dtype='float32') * self._brightness
            light_map = self._unscaled_light_map * color[None, None, :]
            self._light_map = light_map

        return light_map


class Scroll(Item):

    name = "scroll of infinite recursion"
    char = u'次'
    readable = True
    takeable = True
    light_source = False
    mass = 0.05
    length = 20.0
    fg_color = (0.8, 0.8, 0.8, 1.0)

    # page contents shown by the 'read' action (opened as a pager below)
    pages = [
        "Instructions for use:\n"
        "\n"
        "1. Carefully unroll the scroll.\n"
        "2. Read the instructions for use.",

        "3. If the instructions are unclear,\n"
        "   consult the instructions.\n"
        "4. Repeat until enlightened.",

        "You put the scroll down, none the\n"
        "wiser but strangely satisfied.",
    ]

    def read(self, reader):
        """Show the scroll's pages in a modal pager (the joke ends on the last
        page; the scroll is not consumed). Returns the pager session."""
        from . import dialogs
        return dialogs.open_pager(self.scene, self.description, self.pages)


class Torch(Item):
    """A burning torch, which lights its surroundings and flickers as it burns.

    Every torch in the game is flickered by one shared background thread,
    started by the first torch that exists and left running from then on. The
    flame is game state like any other: it advances on its own clock whether or
    not anything is displaying it, and setting ``brightness`` pushes the
    resulting redraw request out through the scene.
    """

    name = "torch"
    char = 't'
    readable = False
    takeable = True
    light_source = True
    mass = 0.5
    length = 30.0
    light_color = (10.0, 8.0, 2.0)
    fg_color = (1.0, 0.8, 0.2, 1.0)

    # Flicker. The flame's log-brightness wanders around 0 as a random walk
    # pulled back to centre (Ornstein-Uhlenbeck), so brightness is log-normal:
    # it never reaches zero, never drifts far, and has no audible period.
    # Working in log units matters because the scene tone-maps lighting with a
    # log as well, so FLICKER_DEPTH lands on screen as a roughly proportional
    # brightness swing rather than being compressed away.
    FLICKER_DEPTH = 1.0     # std dev of log-brightness per kick
    FLICKER_RATE = 4.0      # 1/s; how quickly it wanders (higher = twitchier)
    FLICKER_INTERVAL = 1 / 10.   # seconds between flame updates

    # class default so a torch is well-formed the moment it is created; the
    # flame's state becomes per-instance on its first step
    _log_brightness = 0.0

    # every living torch, weakly held so that dropping one lets it go
    _torches = weakref.WeakSet()
    _flicker_thread = None
    _flicker_lock = threading.Lock()

    def __init__(self, *args, **kwds):
        Item.__init__(self, *args, **kwds)
        with Torch._flicker_lock:
            Torch._torches.add(self)
            if Torch._flicker_thread is None:
                Torch._flicker_thread = threading.Thread(
                    target=Torch._flicker_loop, name='TorchFlicker', daemon=True)
                Torch._flicker_thread.start()

    @staticmethod
    def _flicker_loop():
        """Burn every torch in the game, forever, on this thread."""
        last = time.perf_counter()
        while True:
            time.sleep(Torch.FLICKER_INTERVAL)
            now = time.perf_counter()
            dt, last = now - last, now
            Torch._flicker_all(dt)

    @staticmethod
    def _flicker_all(dt):
        """Advance every visible flame by *dt* seconds, all in one step.

        Torches the player cannot see are skipped: an unwatched flame needs no
        simulating, and skipping it also spares the scene a full-field rescale
        of that torch's light map. The rest advance as a single vectorised
        update, so a tick costs one numpy operation no matter how many torches
        the level holds.
        """
        lit = [t for t in Torch._torches if t.in_player_sight()]
        if not lit:
            return

        # depth and rate are read per torch, so a subclass can burn differently
        log_b = np.array([t._log_brightness for t in lit])
        depth = np.array([t.FLICKER_DEPTH for t in lit])
        rate = np.array([t.FLICKER_RATE for t in lit])

        k = np.minimum(rate * dt, 1.0)
        log_b += k * (np.random.normal(0, 1, size=len(lit)) * depth - log_b)

        for torch, lb in zip(lit, log_b):
            torch._log_brightness = lb
            torch.brightness = float(np.exp(lb))  # pushes redraw + invalidation



