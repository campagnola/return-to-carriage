# coding: utf8
import threading
import time
import weakref

import numpy as np

from .entity import Entity
from .heat import blackbody_color
from .inventory import Inventory
from .light import PointLight
from .location import Location
from .sprite import SingleCharSprite
from .units import K, lm


class Item(Entity):

    name = "nondescript item"
    char = '?'
    mass = 0.0     # in kg
    length = 10.0  # in cm
    readable = False
    takeable = False
    fg_color = (0, 0, 0.8, 1)
    bg_color = None
    #: Light the glyph emits on its own (RGB, linear), or None for a plain
    #: reflective item. An emitter (see Torch) glows independently of the light
    #: falling on it rather than merely reflecting its surroundings.
    emission = None

    def __init__(self, location, scene, obj_name=None):
        Entity.__init__(self, entity_type='item.' + self.name, obj_name=obj_name)
        self.scene = scene

        self.inventory = Inventory(self, allowed_slots=[])
        self.location = Location(self, None, None)
        self.sprite = SingleCharSprite(self, zval=-0.1, char=self.char,
                                       fg_color=self.fg_color, emission=self.emission,
                                       layer='items')

        # A plain item emits no light. An item that shines builds its own
        # Light in its __init__ (see Torch) and drives its colour/brightness;
        # the Light handles the rest of being a light source -- level tracking,
        # shadow maps, the emitted map.
        self.light = None

        scene.add_item(self)

        if location is not None:
            self.location.update(*location)

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
        if self.light is not None:
            self.light.destroy()
        self.location.update(None, None)
        self.sprite.hide()


class Scroll(Item):

    name = "scroll of infinite recursion"
    char = u'次'
    readable = True
    takeable = True
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
    not anything is displaying it, and setting the light's ``brightness`` pushes
    the resulting redraw request out through the scene.

    A torch owns a :class:`~.light.Light`; the torch decides the flame's colour
    and drives its brightness, while the light handles the rest of being a light
    source (level tracking, shadow maps, the emitted map).
    """

    name = "torch"
    char = 't'
    readable = False
    takeable = True
    mass = 0.5
    length = 30.0
    #: Colour temperature of the flame, in kelvin. A wood torch burns low and
    #: sooty -- roughly candle-hot -- so it sits well down the blackbody locus,
    #: deep in the orange. The flame's colour, for both the glyph and the light
    #: it casts, is derived from this rather than hand-picked, so warming or
    #: cooling the flame is a single number (see :func:`~.heat.blackbody_color`).
    FLAME_TEMP = 1800 * K

    #: Peak per-channel luminous flux of the flame, in lumens, that the (0..1)
    #: blackbody hue is scaled up to (see :class:`~.light.PointLight`, which
    #: reads a point light's colour as lumens and paints the lux it produces). A
    #: hand torch is candle-dim -- ~15 lm -- so the player leans on dark
    #: adaptation to see much beyond the next cell. Brightness (flickered around
    #: 1.0) scales it further. Tunable in the visual pass.
    FLAME_FLUX = 15 * lm

    #: Flame colour on the blackbody locus at FLAME_TEMP: an orange whose
    #: brightest channel is ~1, scaled to lumens for the cast light and used
    #: as-is (with full alpha) for the glyph.
    _flame_color = blackbody_color(FLAME_TEMP)
    LIGHT_COLOR = (_flame_color[0] * FLAME_FLUX,
                   _flame_color[1] * FLAME_FLUX,
                   _flame_color[2] * FLAME_FLUX)
    fg_color = _flame_color + (1.0,)
    #: The flame is an emitter: the glyph makes its own light rather than only
    #: reflecting what falls on it. In the same units as the light it casts, so
    #: the glyph glows at the flame's own colour and reads as the source of the
    #: pool of light around it instead of a lit surface the colour of the floor.
    emission = np.array(LIGHT_COLOR) * 3

    # Flicker. The flame's log-brightness wanders around 0 as a random walk
    # pulled back to centre (Ornstein-Uhlenbeck), so brightness is log-normal:
    # it never reaches zero, never drifts far, and has no audible period.
    # Working in log units matters because the scene tone-maps lighting with a
    # log as well, so FLICKER_DEPTH lands on screen as a roughly proportional
    # brightness swing rather than being compressed away.
    FLICKER_DEPTH = 2.0     # std dev of log-brightness per kick
    FLICKER_RATE = 2.0      # 1/s; how quickly it wanders (higher = twitchier)
    FLICKER_INTERVAL = 1 / 10.   # seconds between flame updates

    # class default so a torch is well-formed the moment it is created; the
    # flame's state becomes per-instance on its first step. 1.0 is the nominal
    # flux (the lumens live in LIGHT_COLOR); the flicker wanders around it.
    brightness = 1.0

    # every living torch, weakly held so that dropping one lets it go
    _torches = weakref.WeakSet()
    _flicker_thread = None
    _flicker_lock = threading.Lock()

    def __init__(self, *args, **kwds):
        Item.__init__(self, *args, **kwds)
        self.light = PointLight(self, color=self.LIGHT_COLOR, brightness=self.brightness)
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
        lit = [t for t in Torch._torches if t.light.in_player_sight()]
        if not lit:
            return

        # depth and rate are read per torch, so a subclass can burn differently
        base_brightness = np.array([t.brightness for t in lit])
        depth = np.array([t.FLICKER_DEPTH for t in lit])
        rate = np.array([t.FLICKER_RATE for t in lit])

        new_brightness = base_brightness * depth ** np.random.normal(size=len(lit))
        next_brightness = base_brightness + (new_brightness - base_brightness) * (1 - np.exp(-rate * dt))
        for torch, nb in zip(lit, next_brightness):
            torch.light.brightness = nb



