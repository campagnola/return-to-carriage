"""Eye adaptation: the single, absolute luminance scale the game lives on.

Every level's light is expressed on one shared, physical scale where home
daylight is the reference and the light coming down the hole is a small fraction
of it. Because the scale is absolute, "home is a hundred times brighter than the
sewer" is a fact about the numbers, not an accident of per-level renormalization.
The scale's anchor -- how bright home daylight actually is -- is a level-design
fact and lives with the lights that realise it (see :mod:`.levels`), not here.

What lives here is a model of the viewer's eye. :class:`EyeAdaptation` holds the
one scalar that says how bright the world *currently feels* -- the luminance the
eye has settled to. It is not part of the physical light; it is the observer.
That is why it rides the player (see :class:`~.player.Player`) rather than any
level: it must persist as the player falls through the hole, so the eyes stay
daylight-adapted for the first moment in the dark and only slowly open up.

The eye's reference is not hardcoded: it is *established from the first scene
luminance it is ever shown* -- the first :meth:`~EyeAdaptation.adapt` call, when
adaptation is first needed. Because the player starts in home daylight, that
first measurement is the daylight the eye is adapted to, so "the player starts
adapted to outdoor light" is a measured fact -- exactly consistent with what the
renderer draws -- rather than a stand-in albedo times a magic illuminance. The
dark-adaptation floor (:data:`MIN_ADAPT_LUMINANCE`) is a fixed point on the same
absolute scale, so it does not move with the measured reference.

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

from . import config


#: The darkest reflected luminance the eye will dark-adapt to, in cd/m^2 -- a
#: fixed point on the shared physical scale (not derived from the measured
#: reference). A real eye does not open up without limit: below some level
#: nothing more is gained and true darkness stays dark. This floor caps the
#: exposure (``key / MIN_ADAPT_LUMINANCE`` is the most the tone-mapper will ever
#: brighten a scene), so the dim sewer settles to "the shaft, and not much else"
#: rather than amplifying every corner to grey. It is the main knob for how much
#: is eventually visible in the dark: raise it to keep dark areas darker, lower
#: it to let the eye open up further. Tunable in the visual pass.
MIN_ADAPT_LUMINANCE = 2.0

#: Clamp applied before any ``log``; a truly black scene has zero luminance and
#: ``log(0)`` is undefined, so adaptation targets are floored to this.
_LUMINANCE_FLOOR = 1e-4

#: Adaptation time constants, asymmetric like a real eye. Light adaptation
#: (surroundings brighter than the eye is set for -- the adapted luminance must
#: *rise*, so the screen dims) is quick; dark adaptation (surroundings darker,
#: the adapted luminance *falls* and the screen slowly brightens) is slow. The
#: felt effect: step into glare and the image settles in about a second; drop
#: into the dark and it takes ~20 s for the scene to open up.
TAU_LIGHT_ADAPT = 0.3
TAU_DARK_ADAPT = 8.0

#: Both time constants collapse to this (seconds) under ``config.test_mode``, so
#: the eye settles almost at once and a human testing the game does not wait out
#: the ~20 s dark adaptation. Not used in the real game or the screenshot
#: harness, where the physical time constants above stand.
TAU_TEST = 0.2


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

    def __init__(self, tau_light=TAU_LIGHT_ADAPT, tau_dark=TAU_DARK_ADAPT,
                 key=ADAPT_MIDDLE_GREY, min_luminance=MIN_ADAPT_LUMINANCE):
        """Create an eye with no reference yet.

        The eye holds no adaptation luminance until it is first shown a scene:
        the first :meth:`adapt` (or :meth:`snap_to`) call establishes it from
        the measured scene luminance. Because the player starts in home
        daylight, that first measurement is the daylight the eye is adapted to.
        Until it happens :attr:`luminance`/:attr:`exposure` have no defined value
        and raise if read -- there is nothing to expose to yet.

        :param min_luminance: the darkest luminance the eye will dark-adapt to
            (:data:`MIN_ADAPT_LUMINANCE`) -- a fixed point on the absolute scale,
            so it is known now rather than measured. The adaptation luminance --
            and so the adaptation *target* -- is floored here, capping exposure
            at ``key / min_luminance`` so true darkness stays dark.
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
        if config.test_mode:
            # Hands-on testing: collapse both time constants so the eye settles
            # almost immediately instead of over ~20 s (see TAU_TEST).
            tau_light = tau_dark = TAU_TEST
        self.tau_light = tau_light
        self.tau_dark = tau_dark
        self.key = key
        # Floor on dark adaptation, in the log domain. Kept as a log so it can
        # clamp log_la and the adaptation target directly.
        self.log_min = math.log(max(min_luminance, _LUMINANCE_FLOOR))
        # Log domain (see class docstring): log of the adaptation luminance, so
        # relaxation is constant-stops-per-second. None until the first sight
        # establishes it (see adapt/snap_to); never below the floor thereafter.
        self.log_la = None

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

        The very first call establishes the eye's reference from
        *scene_luminance* -- the eye starts adapted to whatever it first sees --
        and returns without easing: there is no prior state to relax from.
        """
        if self.log_la is None:
            self.snap_to(scene_luminance)
            return
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

        Used where a fully-settled eye is wanted immediately rather than after
        ~20 s: it both establishes the eye's reference on the first sight (the
        no-easing branch of :meth:`adapt`) and serves the deterministic offscreen
        screenshot path, where every capture must be fully adapted to its own
        scene so results do not depend on how many frames were run first.
        """
        self.log_la = max(math.log(max(scene_luminance, _LUMINANCE_FLOOR)), self.log_min)
        self.settling = False

    @property
    def luminance(self):
        """The luminance the eye is currently adapted to (linear, not log).

        Raises before the eye has been shown a scene: there is no reference to
        report until the first :meth:`adapt`/:meth:`snap_to` establishes one.
        """
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
