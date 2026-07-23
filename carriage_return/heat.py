"""Heat: a spot of hot material that glows as it cools back to room temperature.

When a fireball or a bolt of lightning slams into a wall it dumps its energy as
heat (see :mod:`.spell`). We model the aftermath as a :class:`Heat` entity
dropped on the struck cell: it draws no glyph, but it carries a single point
light whose *colour* is the blackbody radiation of its current temperature and
whose *brightness* follows the radiated power. Left alone it cools -- Newton's
law of cooling toward room temperature -- so the light slides down the blackbody
locus (white-hot -> yellow -> orange -> dull red) and dims as it goes, until it
is indistinguishable from the cold wall around it and the entity removes itself.

Like a spell (and a torch's flame) a Heat is *not* part of the turn system: it
ticks on its own daemon thread in real time and pushes each change out through
the level's lighting signal. Its light is pinned to the struck cell rather than
following a host, exactly as a spell pins its lights (see :meth:`.light.Light.pin`).

The temperature is tracked as first-class state -- not merely a colour -- because
a hot cell is useful for more than a glow: later we can let it burn a creature
that steps onto it, ignite a flammable, or be fanned back into a fire. For now
only the light reads it.

Game-side module: no rendering library may be imported here.
"""
import threading
import time

import numpy as np

from .entity import Entity
from .light import PointLight
from .location import Location
from .units import lm


#: Ambient ("room") temperature in kelvin. A cell cools toward this and, once it
#: gets close, has stopped glowing and destroys itself.
AMBIENT_TEMP = 300.0


def blackbody_color(temperature):
    """Approximate the sRGB colour of an ideal blackbody at *temperature* (K).

    Returns an ``(r, g, b)`` tuple of floats in ``0..1`` giving the *hue* of the
    glow only -- the channels are normalised so the brightest is ~1, so this
    says nothing about how bright the source is (that scales as ``T**4``; see
    :meth:`Heat._brightness_for`). Uses Tanner Helland's well-known piecewise fit
    to the Planckian locus, valid roughly 1000-40000 K; below that it tends to a
    deep red, which is the right look for a cooling ember even if it is past the
    fit's calibrated range.
    """
    # the fit is expressed in hundreds of kelvin, clamped to its valid top end
    t = min(float(temperature), 40000.0) / 100.0

    if t <= 66:
        red = 255.0
        # green = 99.4708025861 * np.log(t) - 161.1195681661 if t > 0 else 0.0
        # reduce green just a little
        green = 96 * np.log(t) - 161.1195681661 if t > 0 else 0.0
    else:
        red = 329.698727446 * (t - 60) ** -0.1332047592
        green = 288.1221695283 * (t - 60) ** -0.0755148492

    if t >= 66:
        blue = 255.0
    elif t <= 19:
        blue = 0.0
    else:
        blue = 138.5177312231 * np.log(t - 10) - 305.0447927307

    return tuple(float(np.clip(c, 0.0, 255.0) / 255.0) for c in (red, green, blue))


class Heat(Entity):
    """A glyph-less entity that heats one maze cell and glows while it cools.

    Built on the struck cell with an initial *temperature*; it pins a
    :class:`~.light.PointLight` there whose colour is ``blackbody_color(T)`` and
    whose brightness tracks the radiated power. On its own daemon thread it cools
    toward :data:`AMBIENT_TEMP`, restyling the light each tick, and destroys
    itself once it has cooled to within :data:`DESTROY_MARGIN` of ambient (by
    which point it is no longer glowing). ``advance(dt)`` is factored out and
    takes an injectable ``dt`` so tests can cool it without sleeping, the same
    way :meth:`.spell.Fireball.advance` drives travel.
    """

    #: seconds for the temperature *above ambient* to fall by 1/e (Newton's law
    #: of cooling). A few of these and the glow is gone -- it lingers "a while".
    COOLING_TAU = 4.0
    #: stop and self-destruct once within this many kelvin of ambient
    DESTROY_MARGIN = 10.0
    #: brightness reaches its peak at this temperature and saturates above it,
    #: so a merely-hot fireball strike and a far hotter lightning strike both top
    #: out at a believable glow rather than the bolt washing out the screen.
    REF_TEMP = 3000.0
    #: peak luminous flux of the glow, in lumens, reached at REF_TEMP: the
    #: blackbody colour (0..1 per channel) is scaled by this, so a hot strike
    #: peaks ~10x a torch (a torch is ~15 lm; see :class:`~.light.PointLight`).
    PEAK_BRIGHTNESS = 15000 * lm
    #: real-time cooling tick
    STEP_INTERVAL = 1 / 30.

    def __init__(self, scene, maze, pos, temperature, start=True):
        Entity.__init__(self, entity_type='mob.heat', obj_name='hot spot')
        self.scene = scene
        self.maze = maze
        self.pos = tuple(int(v) for v in pos)
        self.temperature = float(temperature)
        self._done = False
        self._thread = None

        # An empty location for the light to hang off; like a spell, the heat
        # pins its light to a cell directly and never enters the maze inventory,
        # so it stays invisible to walkability / on_walked_on.
        self.location = Location(self, None, None)
        self.light = PointLight(self, color=(1.0, 1.0, 1.0), brightness=0.0)
        self.light.pin(self.maze, self.pos)

        self._restyle()          # colour + brightness for the starting temp
        self._relight()          # nudge the level to composite the new light
        if start:
            self._start()

    # -- temperature -> light ---------------------------------------------

    def _brightness_for(self, temperature):
        """Point-light brightness for *temperature*, following radiated power.

        Radiated power goes as ``T**4`` (Stefan-Boltzmann), referenced to the
        ambient floor so a cell at room temperature glows not at all, and
        normalised to :data:`REF_TEMP` then clamped to :data:`PEAK_BRIGHTNESS`.
        """
        num = temperature ** 4 - AMBIENT_TEMP ** 4
        den = self.REF_TEMP ** 4 - AMBIENT_TEMP ** 4
        return self.PEAK_BRIGHTNESS * float(np.clip(num / den, 0.0, 1.0))

    def _restyle(self):
        """Set the light's colour and brightness from the current temperature."""
        self.light.color = blackbody_color(self.temperature)
        self.light.brightness = self._brightness_for(self.temperature)

    def _relight(self):
        """Recomposite this cell's level and ask for a repaint.

        Pinning a light does not on its own mark the level's lighting stale;
        firing the light's ``changed`` signal drives the level exactly as the
        torch flicker thread and the spells do.
        """
        self.light.changed()

    # -- cooling ----------------------------------------------------------

    def advance(self, dt):
        """Cool for *dt* seconds and restyle the glow; True while still hot.

        Newton's law of cooling gives an exact exponential relaxation toward
        ambient over the step, so the result is independent of tick size and a
        long test step matches many short real ones. Destroys the entity and
        returns False once it has cooled to within :data:`DESTROY_MARGIN` of
        ambient. Injectable ``dt`` so tests drive cooling without sleeping.
        """
        above = self.temperature - AMBIENT_TEMP
        self.temperature = AMBIENT_TEMP + above * np.exp(-dt / self.COOLING_TAU)
        if self.temperature - AMBIENT_TEMP <= self.DESTROY_MARGIN:
            self.destroy()
            return False
        self._restyle()
        return not self._done

    def _start(self):
        self._thread = threading.Thread(target=self._run, name=str(self.type),
                                        daemon=True)
        self._thread.start()

    def _run(self):
        last = time.perf_counter()
        while not self._done:
            time.sleep(self.STEP_INTERVAL)
            now = time.perf_counter()
            dt, last = now - last, now
            self.advance(dt)

    def destroy(self):
        """Cool spot gone: light out, level repainted. Idempotent.

        May be reached from the cooling thread (natural end) or an external
        caller; like a spell it only mutates game-side state and fires
        thread-safe signals, so it is safe on whatever thread calls it.
        """
        if self._done:
            return
        self._done = True
        self.light.destroy()
        level = self.maze.level
        if level is not None:
            level.invalidate_lighting()
            level.lighting_changed()
