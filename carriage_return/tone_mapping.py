"""The full tone-reproduction chain: albedo + emission + illuminance + eye
adaptation -> the color a pixel is actually drawn.

Every level's light is expressed on one shared, physical scale where home
daylight is the reference and the light coming down the hole is a small
fraction of it. Because the scale is absolute, "home is a hundred times
brighter than the sewer" is a fact about the numbers, not an accident of
per-level renormalization. The scale's anchor -- how bright home daylight
actually is -- is a level-design fact and lives with the lights that realise
it (see :mod:`.levels`), not here.

This module holds everything downstream of that light: the eye's adaptation
state (:class:`EyeAdaptation`), the exposure it produces, and the Reinhard +
display-gamma curve that turns exposed linear luminance into a display pixel.
Two call sites need the same curve -- the live scene, which the GPU draws
(:class:`~.backends.vispy.graphics.TextureMaskFilter`), and the memory
overlay, which the CPU draws (:meth:`~.world.Level.update_sight`) -- so the
curve and its constants (:data:`DISPLAY_GAMMA`) are defined once, here, and
both call sites use them. Where the same formula must also run on the GPU,
its GLSL twin is defined right next to the Python version below (see
:data:`GLSL_REINHARD_TONEMAP`) -- edit both together, they are not allowed to
drift apart.

Game-side module: no rendering library may be imported here. Only ``math``
and numpy are used.
"""
import math

import numpy as np

from . import config


# ---------------------------------------------------------------------------
# Illuminance -> luminance
# ---------------------------------------------------------------------------

#: Rec. 709 luminance weights. Used to collapse an RGB value to a single
#: perceived brightness -- both for the eye-adaptation target and for the
#: per-cell material reflectance the albedo map holds.
LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype='float32')


# ---------------------------------------------------------------------------
# Display gamma
# ---------------------------------------------------------------------------

#: The display transfer function exponent applied after every Reinhard curve
#: in the game, on both the CPU and GPU paths: the live scene
#: (:data:`GLSL_REINHARD_TONEMAP`, run by TextureMaskFilter) and the memory
#: overlay (:func:`memory_overlay_pixel`, run by Level.update_sight) both
#: gamma-correct with ``x ** (1 / DISPLAY_GAMMA)``. One constant so the two
#: paths cannot drift apart. Currently mid-experiment on the middle-grey key
#: (see :data:`ADAPT_MIDDLE_GREY`); the photographic display standard is 2.2.
DISPLAY_GAMMA = 2.2


def reinhard_tonemap(exposed):
    """Reinhard tone curve + display gamma.

    *exposed* is linear HDR luminance already multiplied by exposure (see
    :func:`exposure`); returns display-ready values in [0, 1]. The GLSL twin
    of this function, run per-fragment on the GPU, is
    :data:`GLSL_REINHARD_TONEMAP` -- keep the two in sync.
    """
    exposed = np.asarray(exposed, dtype=float)
    lit = exposed / (1.0 + exposed)
    return lit ** (1.0 / DISPLAY_GAMMA)


#: GLSL source for a ``reinhard_tonemap`` dependency function implementing
#: exactly the math :func:`reinhard_tonemap` does above, for
#: :class:`~.backends.vispy.graphics.TextureMaskFilter` to call from its
#: fragment shader. ``DISPLAY_GAMMA`` is baked in at import time, so the two
#: functions cannot disagree about which gamma they used -- edit the constant
#: above, not the number below.
GLSL_REINHARD_TONEMAP = """
vec3 reinhard_tonemap(vec3 exposed) {{
    vec3 lit = exposed / (1.0 + exposed);
    return pow(lit, vec3(1.0 / {gamma}));
}}
""".format(gamma=DISPLAY_GAMMA)


# ---------------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------------

def exposure(key, adapted_luminance):
    """The Reinhard exposure scalar: ``key / adapted_luminance``.

    The brighter the reference (the eye's adaptation luminance for the live
    scene, :data:`MEMORY_REF_LUMINANCE` for the memory overlay), the smaller
    this is, so the same physical light is drawn darker.
    """
    return key / adapted_luminance


def reflected_luminance(albedo, illuminance):
    """Linear reflected luminance leaving a surface: ``albedo * illuminance``.

    This is the literal quantity the GPU fragment shader computes for the
    live scene (no BRDF normalization) -- gating by line of sight and adding
    emission is the caller's job (see
    :class:`~.backends.vispy.graphics.TextureMaskFilter`). Contrast
    :func:`scene_reflected_luminance`, which *does* apply the Lambertian
    1/pi, for the physically-estimated luminance the eye adapts to.
    """
    return np.asarray(albedo, dtype=float) * np.asarray(illuminance, dtype=float)


def scene_reflected_luminance(albedo_luminance, illuminance_luminance):
    """Reflected luminance (cd/m^2) of a Lambertian surface, for eye adaptation.

    A Lambertian surface of reflectance ``rho`` lit by ``E`` lux has luminance
    leaving it of ``rho * E / pi``; the 1/pi turns arriving light into light
    leaving the surface toward the eye. Used to build the adaptation target
    the eye samples (:meth:`~.world.Level.update_sight`) -- *not* the same
    quantity :func:`reflected_luminance` computes for the actual displayed
    pixel, which omits the 1/pi.
    """
    return albedo_luminance * illuminance_luminance / math.pi


def pixel_color(albedo, emission, illuminance, exposure_value):
    """albedo, emission, illuminance, exposure -> displayed rgb, per channel.

    The exact math :class:`~.backends.vispy.graphics.TextureMaskFilter` runs
    per-fragment on the GPU, reproduced in Python for tools (see
    ``agent_helpers/tonemap_demo.ipynb``) that want to predict what a color
    will look like on screen without a GL context.
    """
    refl = reflected_luminance(albedo, illuminance)
    out_lum = refl + np.asarray(emission, dtype=float)
    return reinhard_tonemap(out_lum * exposure_value)


# ---------------------------------------------------------------------------
# Memory overlay
# ---------------------------------------------------------------------------

#: The middle-grey key the memory overlay is exposed at (see :func:`exposure`),
#: a *fixed* reference deliberately independent of the player's live
#: adaptation (:data:`ADAPT_MIDDLE_GREY`): a remembered area is a faint
#: recollection, not something that should brighten just because the eye is
#: now dark-adapted.
MEMORY_KEY = 0.18

#: Reference luminance the memory overlay is exposed against (cd/m^2, same
#: scale as the eye's adaptation). Paired with :data:`MEMORY_KEY` via
#: :func:`exposure` to get the memory overlay's fixed exposure.
MEMORY_REF_LUMINANCE = 0.64

#: Caps how bright the memory overlay's recollection gets on screen.
MEMORY_STRENGTH = 1.0

_MEMORY_EXPOSURE = exposure(MEMORY_KEY, MEMORY_REF_LUMINANCE)


def memory_overlay_pixel(memory_luminance):
    """Remembered linear luminance -> display-space memory overlay value.

    Reinhard + display gamma under the fixed :data:`MEMORY_KEY` /
    :data:`MEMORY_REF_LUMINANCE` exposure (see :func:`exposure`), scaled by
    :data:`MEMORY_STRENGTH`. Used by :meth:`~.world.Level.update_sight` to
    paint ``memory_overlay``.
    """
    exposed = np.asarray(memory_luminance, dtype=float) * _MEMORY_EXPOSURE
    return reinhard_tonemap(exposed) * MEMORY_STRENGTH


# ---------------------------------------------------------------------------
# Eye adaptation
# ---------------------------------------------------------------------------

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

#: The brightest reflected luminance the eye will light-adapt to, in cd/m^2 --
#: the ceiling to :data:`MIN_ADAPT_LUMINANCE`'s floor, and likewise a fixed point
#: on the shared physical scale. A real eye cannot stop down without limit
#: either: past some level it can no longer compress the light and an intense
#: source just glares. This ceiling floors the exposure (``key /
#: MAX_ADAPT_LUMINANCE`` is the least the tone-mapper will ever brighten a
#: scene), so a surface lit fiercely up close -- a torch or fireball at arm's
#: length -- stays a blown-out glare instead of the eye adapting it down to grey.
#: Set above home daylight (the game's brightest illuminance, and the eye's own
#: established reference) so ordinary daylight adapts fully and only genuinely
#: intense light clips. Tunable in the visual pass.
MAX_ADAPT_LUMINANCE = 20000.0

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


#: The middle-grey key value for the Reinhard exposure (see :func:`exposure`).
#: 0.18 is photographic 18% grey; lower draws the whole scene darker. It sets
#: what adapted luminance is mapped to mid-tone on screen. Currently
#: mid-experiment; see :data:`DISPLAY_GAMMA`.
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
                 key=ADAPT_MIDDLE_GREY, min_luminance=MIN_ADAPT_LUMINANCE,
                 max_luminance=MAX_ADAPT_LUMINANCE):
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
        :param max_luminance: the brightest luminance the eye will light-adapt to
            (:data:`MAX_ADAPT_LUMINANCE`) -- the matching ceiling, likewise a
            fixed point on the absolute scale. The adaptation luminance and target
            are capped here, flooring exposure at ``key / max_luminance`` so an
            intense source stays a glare rather than adapting down to grey.
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
        # Floor and ceiling on adaptation, in the log domain (set as logs so they
        # can clamp log_la and the adaptation target directly). A level may
        # retune these as the player enters it; see set_bounds.
        self.set_bounds(min_luminance, max_luminance)
        # Log domain (see class docstring): log of the adaptation luminance, so
        # relaxation is constant-stops-per-second. None until the first sight
        # establishes it (see adapt/snap_to); never below the floor thereafter.
        self.log_la = None

        #: True while the eye is still more than _SETTLE_EPS (log units) from
        #: its target. The renderer reads this to keep repainting an otherwise
        #: static scene until adaptation has caught up, then lets it go idle.
        self.settling = False

    def set_bounds(self, min_luminance=None, max_luminance=None):
        """Set the range of luminance the eye is allowed to adapt to.

        The bounds are level design, not eye state: which level the player is on
        decides how far the eye may open up or stop down (see
        :meth:`~.world.Level.update_sight`). ``None`` for either bound resets it
        to the module default (:data:`MIN_ADAPT_LUMINANCE` /
        :data:`MAX_ADAPT_LUMINANCE`), which is what a level that says nothing
        about adaptation gets. The eye's *state* -- the luminance it has settled
        to -- is untouched here; only the range it clamps into changes, so the
        eye still eases across a bound change rather than jumping.

        Passing the **same** value for both pins the eye: every target and every
        adapted luminance clamps to that one value, so adaptation stops sampling
        the scene entirely and the exposure is fixed. A level uses this when the
        scene it draws is a poor guide to how bright it should feel -- home, lit
        by the sky rather than the dim floor the sampling window sees.
        """
        if min_luminance is None:
            min_luminance = MIN_ADAPT_LUMINANCE
        if max_luminance is None:
            max_luminance = MAX_ADAPT_LUMINANCE
        self.log_min = math.log(max(min_luminance, _LUMINANCE_FLOOR))
        self.log_max = math.log(max(max_luminance, _LUMINANCE_FLOOR))

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
        # Clamp the target into the adaptation range, not just log_la after the
        # fact: a scene outside [floor, ceiling] is unreachable, and measuring
        # settling against an unreachable target would never clear the flag --
        # the renderer would repaint at 60 Hz forever.
        target = min(max(math.log(max(scene_luminance, _LUMINANCE_FLOOR)),
                         self.log_min), self.log_max)
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
        self.log_la = min(max(math.log(max(scene_luminance, _LUMINANCE_FLOOR)),
                              self.log_min), self.log_max)
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
        return exposure(self.key, self.luminance)
