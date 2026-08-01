"""Perception: the shared aspect of anything the player may notice.

Where the rest of the game asks *what is true* -- a cell is dark, a mob is
invisible, hunger is rising -- perception asks the separate question *does the
player notice it, and how loudly*. Keeping that question in one place lets a
darkness warning, a faint noise, and a hidden button all resolve the same way:
each is a Percept, and the scene's resolver (see Scene.perceive) rolls the
perceiver's acuity against the percept's detectability, then either speaks up
now, holds the observation for a later "look", or stays silent.

Two logarithmic scales, both centred on zero, run through everything here:

- *salience* is how much a percept matters: 0 is maximally cared about, and
  negative values matter progressively less. It decides auto-log vs held.
- *visibility* is how easy a percept is to detect: 0 is certainly noticed, and
  negative values are progressively harder to catch. The perceiver's perception
  stat is added to it at resolve time, so a keener eye lifts every faint thing
  toward the threshold at once.

A Percept is deliberately *not* an Entity. Persistent world objects (a mob, a
button, a wall) are their own class family and stay so; what they share with a
transient message ("you hear a noise") is only the thin perceivable aspect,
which is this. An Entity that wants to be separately perceivable hangs a Percept
off ``entity.percepts`` (see entity.py) rather than becoming one.
"""
import math

from .events import Observable


#: Salience at or above which a percept is logged the instant it is noticed,
#: rather than held for a later "look". The boundary sits at 0 -- "maximally
#: cared about" -- so anything with non-negative salience speaks up immediately
#: and only the explicitly quieter (negative) percepts wait to be looked for.
AUTO_LOG_SALIENCE = 0.0


class Percept:
    """The shared aspect of something the player may perceive.

    A Percept knows how much it matters (salience), how hard it is to notice
    (visibility), how often it may fire again (cooldown), and what to say when
    noticed (description). Deciding *whether* it is noticed on a given occasion
    is not its job -- that belongs to the scene's perception resolver
    (Scene.perceive), which weighs the perceiver's acuity against this percept's
    visibility.
    """
    def __init__(self, description:str, salience:float, visibility:float, cooldown:float=0.0, sense:str='generic'):
        #: log-scale importance; higher matters more (0 ~ maximally cared about,
        #: negative = progressively less). Decides auto-log vs held-for-"look".
        self.salience = salience
        #: log-scale detectability; higher is easier to notice (0 ~ certainly
        #: noticed, negative = progressively harder). The perceiver's perception
        #: stat is added to this when the scene resolves a sighting.
        self.visibility = visibility
        #: wall-clock seconds that must elapse before this may be perceived again.
        self.cooldown = cooldown
        #: a str or a Description; describe() renders it for a given perceiver.
        self.description = description
        #: monotonic time this was last perceived, or None if never.
        self.last_perceived = None
        #: fired as ``perceived(who=<perceiver>)`` once this is perceived, so
        #: other systems can react without the resolver knowing about them.
        self.perceived = Observable(source=self)
        #: a short tag naming what channel this is perceived through ('vision',
        #: 'hearing', ...) -- purely descriptive metadata a perceiver can key
        #: its own per-sense acuity off of; carries no behavior on its own.
        self.sense = sense

    def ready(self, now):
        """True if the cooldown has elapsed since this was last perceived.

        *now* is a monotonic clock reading (see Scene.clock). A percept never
        yet perceived is always ready.
        """
        return self.last_perceived is None or (now - self.last_perceived) >= self.cooldown

    def mark(self, now):
        """Record that this was perceived at monotonic time *now*.

        Opens the cooldown window: ``ready`` stays False until it elapses.
        """
        self.last_perceived = now

    def detectability(self, perceiver=None):
        """{sense: value} -- how physically detectable this is to a generic
        observer at *perceiver*'s position: lighting, camouflage, distance,
        whatever this percept's sense cares about. Perceiver-specific stats
        (acuity, adaptation, spells) do NOT belong here -- Scene.perceive
        weighs those afterward, through the perceiver itself. Default ignores
        *perceiver* and returns the fixed `visibility` set at construction
        under this percept's `sense`, for percepts with no position at all
        (the grue's ambient warning)."""
        return {self.sense: self.visibility}

    def describe(self, perceiver=None):
        """The text *perceiver* reads on noticing this.

        Named to avoid clashing with the stored ``description``. Returns the
        stored text; the ``perceiver`` argument is the seam where
        perceiver-dependent detail (a keener eye reading more into the same
        percept) will eventually grow.
        """
        return str(self.description)

    def on_perceived(self, scene, who):
        """Default effect when this percept is auto-logged: say it, announce it.

        Writes the description to the scene's message log and fires
        ``perceived``. Subclasses override to add effects (a trap that springs,
        a mob that reacts to being spotted) while reproducing the log-and-fire.
        """
        scene.write(self.describe(who))
        self.perceived(who=who)


class TransientPercept(Percept):
    """A discrete one-shot event: it happens once, and a player who misses it
    misses it.

    "You hear a noise", "you fall into a hole", "it is dark here" -- a momentary
    occurrence with no lasting state to re-observe. If the perception roll fails
    it is simply gone; there is nothing left to look at. A cooldown keeps a
    repeating trigger (shuffling around in the dark) from spamming the log: the
    same warning fires again only once its window has elapsed. It needs nothing
    beyond Percept today; it exists to name the intent and to be the type that
    future one-shot code matches against. (The grue uses this with a cooldown.)
    """


class ContinuousPercept(Percept):
    """A percept whose value varies continuously (hunger, detection of magic).

    Unlike a one-shot TransientPercept, this tracks a live value, and both of
    perception's scales can in principle be derived from it: a *rapid* change is
    more visible than a slow drift (you notice the moment hunger bites, not each
    imperceptible increment), and particular values are more salient than others
    (a full belly is unremarkable; starvation demands attention). Its
    description likewise depends on the current value -- "peckish" vs
    "ravenous". No concrete client exists yet, so this is only the skeleton:
    ``update`` records the latest value and is the documented seam where a
    future version would raise ``visibility`` from the rate of change, and
    ``describe`` is overridable to choose text by value.
    """
    def __init__(self, description, salience, visibility, cooldown=0.0):
        super().__init__(description, salience, visibility, cooldown=cooldown)
        #: the latest value fed in through update(); None until first set.
        self.value = None

    def update(self, value):
        """Record a new current *value*.

        The seam for value-driven perception: a future version would compare it
        with the previous value and raise ``visibility`` when it changes fast.
        For now it only stores the value so ``describe`` can key off it.
        """
        self.value = value

    def describe(self, perceiver=None):
        """Value-dependent text; falls back to the base description for now."""
        return super().describe(perceiver)


class VisualPercept(Percept):
    """A percept detected by sight, computed from illuminance at an entity's
    own cell and distance to the observer.

    `visibility` (base class) sets how easy *this particular* percept is to
    trigger relative to the shared light/distance calculation below -- a
    higher value needs less light or less proximity to clear. Two percepts on
    the same entity that differ only in how easy they are to catch (a glint
    vs. a full look) should both be plain VisualPercepts with different
    `visibility`, not separate subclasses -- write a subclass only for
    something that actually computes detectability differently.

    No separate line-of-sight check is needed here: a point light's
    composited illuminance is already gated by the shadow map that casts it
    (see PointLight._render_light_map in light.py), so an occluded cell
    already reads as dark without this needing to ask twice. A dedicated
    line-of-sight field does exist on Level, but it is only ever populated by
    the real GPU shadow-casting draw thread -- gating on it here would make
    every VisualPercept silently undetectable in every headless test (and
    this codebase's test suite is deliberately headless; see
    tests/test_perception.py).
    """

    #: illuminance (lux) above which the reflected-light term saturates at
    #: `visibility`; controls how quickly detectability rises with light.
    #: Tunable in the visual pass.
    DIFFUSE_SCALE = 2.0
    #: cells; floor under the distance used for the 1/r^2-equivalent falloff,
    #: so standing on the entity's own cell does not blow up.
    MIN_DISTANCE = 1.0

    def __init__(self, description, salience, visibility, glow=0.0, entity=None, cooldown=0.0):
        super().__init__(description, salience, visibility, cooldown=cooldown, sense='vision')
        #: log-scale self-emission strength: how detectable this is by its own
        #: glow alone in complete darkness. 0 (default) means no glow -- only
        #: reflected light can ever reveal it.
        self.glow = glow
        #: the Entity (e.g. an Item) this percept is attached to; its location
        #: and the level it's on are read fresh on every check.
        self.entity = entity

    def detectability(self, perceiver=None):
        loc = self.entity.location.global_location
        if loc is None:
            return {'vision': float('-inf')}
        level = loc.container.level

        luminance = level.luminance_at(loc.slot)
        if luminance is None:
            return {'vision': float('-inf')}

        # reflected light: rises toward `visibility` as the cell brightens,
        # saturating there once luminance clears DIFFUSE_SCALE -- more light
        # past "enough" doesn't make it any easier to make out. True darkness
        # (luminance <= 0) is undetectable outright, regardless of this
        # percept's own visibility or the perceiver's acuity.
        if luminance <= 0:
            diffuse_vis = float('-inf')
        else:
            diffuse_vis = self.visibility + min(0.0, math.log10(luminance / self.DIFFUSE_SCALE))

        # self-glow: strongest in the dark, washed out as ambient light rises
        # past the glow's own brightness -- a faint glow disappears in
        # daylight, the way a firefly is invisible at noon.
        if self.glow > 0:
            emissive_vis = self.glow * math.exp(-luminance / self.glow)
        else:
            emissive_vis = float('-inf')

        vis = max(diffuse_vis, emissive_vis)

        p_loc = perceiver.location.global_location
        if p_loc is None:
            return {'vision': float('-inf')}
        distance = math.dist(loc.slot, p_loc.slot)
        # inverse-square falloff, expressed additively in this log-scale
        # system: log10(1/r^2) == -2*log10(r).
        return {'vision': vis - 2 * math.log10(max(distance, self.MIN_DISTANCE))}
