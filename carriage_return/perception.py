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
off ``entity.percept`` (see entity.py) rather than becoming one.
"""
from .events import Observable


class Percept:
    """The shared aspect of something the player may perceive.

    A Percept knows how much it matters (salience), how hard it is to notice
    (visibility), how often it may fire again (cooldown), and what to say when
    noticed (description). Deciding *whether* it is noticed on a given occasion
    is not its job -- that belongs to the scene's perception resolver
    (Scene.perceive), which weighs the perceiver's acuity against this percept's
    visibility.
    """
    def __init__(self, description:str, salience:float, visibility:float, cooldown:float=0.0):
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
