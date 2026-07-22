"""Eye adaptation: the single, absolute luminance scale the game lives on.

Every level's light is expressed on one shared, physical scale where home
daylight is the reference and the light coming down the hole is a small fraction
of it. Because the scale is absolute, "home is a hundred times brighter than the
sewer" is a fact about the numbers, not an accident of per-level renormalization.
Two things follow from an absolute scale, and both live here:

- A reference point for that scale. :data:`OUTDOOR_ILLUMINANCE` and
  :data:`REFERENCE_ALBEDO` are the one place the daylight reference is written
  down; :data:`OUTDOOR_ADAPT_LUMINANCE` is the reflected luminance the eye is
  adapted to when standing in open daylight. Other modules (the level builders,
  the default :class:`EyeAdaptation`) import these constants rather than
  spelling out magic numbers, so "the player starts adapted to outdoor light"
  and "home ambient is this bright" cannot drift apart.

- A model of the viewer's eye. :class:`EyeAdaptation` holds the one scalar that
  says how bright the world *currently feels* -- the luminance the eye has
  settled to. It is not part of the physical light; it is the observer. That is
  why it rides the player (see :class:`~.player.Player`) rather than any level:
  it must persist as the player falls through the hole, so the eyes stay
  daylight-adapted for the first moment in the dark and only slowly open up.

This module is the CPU half of the tone-reproduction chain. It produces one
number -- :attr:`EyeAdaptation.exposure` -- that the GPU Reinhard tone-mapper
consumes as a uniform. The fragment shader reconstructs reflected luminance and
maps it to the display as::

    v   = exposure * albedo * illuminance      # exposed reflected luminance
    lit = v / (1 + v)                          # Reinhard
    out = lit ** (1 / 2.2)                      # display gamma

so this module never touches a pixel; it only decides where middle grey sits.

Game-side module: no rendering library may be imported here. Only ``math`` (and
numpy, were it needed) is used.
"""
import math


#: Home daylight illuminance, per channel, in arbitrary "lux-like" units. This
#: is the reference the whole light scale is pinned to; every other light in the
#: game is chosen relative to it (the hole shaft is ~1/100 of this).
OUTDOOR_ILLUMINANCE = 100.0

#: A representative material reflectance, used only to relate an illuminance
#: (light arriving at a surface) to a luminance (light leaving it toward the
#: eye). Real surfaces vary; this stands in for a "typical" one so the eye's
#: adaptation target has a physical meaning without needing a real albedo map.
REFERENCE_ALBEDO = 0.15

#: The reflected luminance the eye is adapted to outdoors -- daylight bounced off
#: a representative surface. This is the default the eye starts at, so the player
#: begins fully adapted to open daylight.
OUTDOOR_ADAPT_LUMINANCE = OUTDOOR_ILLUMINANCE * REFERENCE_ALBEDO

#: The darkest luminance the eye will dark-adapt to. A real eye does not open up
#: without limit: below some level nothing more is gained and true darkness
#: stays dark. This floor caps the exposure (``key / MIN_ADAPT_LUMINANCE`` is the
#: most the tone-mapper will ever brighten a scene), so the dim sewer settles to
#: "the shaft, and not much else" rather than amplifying every corner to grey.
#: It is the main knob for how much is eventually visible in the dark: raise it
#: to keep dark areas darker, lower it to let the eye open up further.
MIN_ADAPT_LUMINANCE = OUTDOOR_ADAPT_LUMINANCE / 50.0

#: Clamp applied before any ``log``; a truly black scene has zero luminance and
#: ``log(0)`` is undefined, so adaptation targets are floored to this.
_LUMINANCE_FLOOR = 1e-4

#: Adaptation time constants, asymmetric like a real eye. Light adaptation
#: (surroundings brighter than the eye is set for -- the adapted luminance must
#: *rise*, so the screen dims) is quick; dark adaptation (surroundings darker,
#: the adapted luminance *falls* and the screen slowly brightens) is slow. The
#: felt effect: step into glare and the image settles in about a second; drop
#: into the dark and it takes ~20 s for the scene to open up.
TAU_LIGHT_ADAPT = 1.0
TAU_DARK_ADAPT = 2.0


#: The middle-grey key value for the Reinhard exposure. 0.18 is photographic
#: 18% grey; lower draws the whole scene darker. It sets what adapted luminance
#: is mapped to mid-tone on screen.
ADAPT_MIDDLE_GREY = 0.08


#: Below this log-luminance gap the eye counts as settled. The renderer stops
#: forcing repaints once adaptation is within this of its target, so a static
#: scene goes idle again instead of animating an imperceptible last sliver.
_SETTLE_EPS = 0.01


class EyeAdaptation:
    """The viewer's eye, modelled as a single luminance it has settled to.

    One scalar of state: the luminance the world currently *feels* like, which
    sets where middle grey lands and therefore how bright everything is drawn.
    It relaxes toward whatever the player is actually looking at, so walking
    from daylight into a dark sewer leaves the eye briefly over-adapted (the
    surroundings look black) and it opens up over the adaptation time constant.

    The state is kept in the **log domain** (:attr:`log_la`, the log of the
    adaptation luminance). Adaptation felt in log luminance is adaptation in
    *stops*: closing a fixed fraction of the log-distance each second means a
    hundredfold drop in light takes the same felt time to adjust to as a
    tenfold one, regardless of the absolute levels involved. That is how real
    eyes behave. Adaptation is also **asymmetric**: brightening the eye to a
    lighter scene (:data:`TAU_LIGHT_ADAPT`, stepping into glare) is much faster
    than opening it up in the dark (:data:`TAU_DARK_ADAPT`), so :meth:`adapt`
    picks the time constant by which way it is moving.

    **Threading -- single owner, no lock.** This object is thread-unsafe by
    design. Exactly one thread (the draw thread, inside ``Level.update_sight``)
    ever calls :meth:`adapt`/:meth:`snap_to`, and the same thread reads
    :attr:`exposure` the same frame. There is no cross-thread sharing to guard,
    so there is no lock; adding one would only signal a sharing that must not
    happen. (Consistent with the project's lock-free, single-owner model.)

    Pairs with the GPU Reinhard tone-mapper: :attr:`exposure` is the scalar the
    shader multiplies reflected luminance by before the Reinhard curve.
    """

    def __init__(self, luminance=OUTDOOR_ADAPT_LUMINANCE,
                 tau_light=TAU_LIGHT_ADAPT, tau_dark=TAU_DARK_ADAPT, key=ADAPT_MIDDLE_GREY,
                 min_luminance=MIN_ADAPT_LUMINANCE):
        """Start the eye adapted to *luminance*.

        :param luminance: the reflected luminance the eye begins settled to.
            Defaults to :data:`OUTDOOR_ADAPT_LUMINANCE`, so a freshly built eye
            is fully daylight-adapted -- the correct state for a player who has
            not yet fallen down the hole.
        :param min_luminance: the darkest luminance the eye will dark-adapt to
            (:data:`MIN_ADAPT_LUMINANCE`). The adaptation luminance -- and so the
            adaptation *target* -- is floored here, capping exposure at
            ``key / min_luminance`` so true darkness stays dark.
        :param tau_light: light-adaptation time constant in seconds (~1 s), used
            when the surroundings are *brighter* than the eye is set for so the
            adapted luminance rises and the screen dims. The eye closes
            ``1 - exp(-dt/tau)`` of the remaining log-distance each step.
        :param tau_dark: dark-adaptation time constant in seconds (~20 s), used
            when the surroundings are *darker* so the adapted luminance falls and
            the screen slowly brightens.
        :param key: the middle-grey key value for the Reinhard exposure (0.18 is
            photographic 18% grey; lower draws the whole scene darker). It sets
            what adapted luminance is mapped to mid-tone on screen.
        """
        self.tau_light = tau_light
        self.tau_dark = tau_dark
        self.key = key
        # Floor on dark adaptation, in the log domain. Kept as a log so it can
        # clamp log_la and the adaptation target directly.
        self.log_min = math.log(max(min_luminance, _LUMINANCE_FLOOR))
        # Log domain (see class docstring): store log of the adaptation
        # luminance so relaxation is constant-stops-per-second. Never below the
        # dark-adaptation floor.
        self.log_la = max(math.log(max(luminance, _LUMINANCE_FLOOR)), self.log_min)

        #: True while the eye is still more than _SETTLE_EPS (log units) from
        #: its target. The renderer reads this to keep repainting an otherwise
        #: static scene until adaptation has caught up, then lets it go idle.
        self.settling = False

    def adapt(self, scene_luminance, dt):
        """Relax the eye toward *scene_luminance* over *dt* seconds.

        The move happens in the log domain::

            target      = log(max(scene_luminance, FLOOR))
            self.log_la += (target - self.log_la) * (1 - exp(-dt / tau))

        a discrete exponential approach, framerate-independent for any ``dt``.
        The time constant is chosen by direction: :attr:`tau_light` when the eye
        must brighten-adapt to a lighter scene (``target`` above the current
        level), :attr:`tau_dark` when it must dark-adapt to a darker one. A
        non-positive ``dt`` is a no-op (no time has passed), which also makes a
        paused or single-shot frame safe to feed in.

        Also updates :attr:`settling` to whether the eye is still more than
        ``_SETTLE_EPS`` from its target, so the renderer can keep the frame loop
        alive until adaptation has caught up even when nothing else is changing.
        """
        if dt <= 0:
            return
        # Clamp the target to the dark-adaptation floor, not just log_la after
        # the fact: a scene darker than the floor is unreachable, and measuring
        # settling against an unreachable target would never clear the flag --
        # the renderer would repaint at 60 Hz forever.
        target = max(math.log(max(scene_luminance, _LUMINANCE_FLOOR)), self.log_min)
        gap = target - self.log_la
        self.settling = abs(gap) > _SETTLE_EPS
        tau = self.tau_light if gap > 0 else self.tau_dark
        self.log_la += gap * (1.0 - math.exp(-dt / tau))

    def snap_to(self, scene_luminance):
        """Instantly adapt the eye to *scene_luminance* -- no easing.

        Used where a settled eye is wanted immediately rather than after ~20 s:
        initialising the player at game start, and the deterministic offscreen
        screenshot path, where every capture must be fully adapted to its own
        scene so results do not depend on how many frames were run first.
        """
        self.log_la = max(math.log(max(scene_luminance, _LUMINANCE_FLOOR)), self.log_min)
        self.settling = False

    @property
    def luminance(self):
        """The luminance the eye is currently adapted to (linear, not log)."""
        return math.exp(self.log_la)

    @property
    def exposure(self):
        """The exposure scalar the GPU tone-mapper multiplies light by.

        ``key / adapted_luminance``: the brighter the eye has adapted, the
        smaller this is, so the same physical light is drawn darker -- the eye
        stopping down in bright surroundings. The shader uses it as the
        ``exposure`` in ``v = exposure * albedo * illuminance`` before the
        Reinhard curve.
        """
        return self.key / self.luminance
