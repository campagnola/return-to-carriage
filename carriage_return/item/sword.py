# coding: utf8
"""The :class:`Sword` item: a rusty sword you notice differently by light."""
from ..perception import VisualPercept
from .base import Item

#: How easily each of the sword's percepts triggers, in the same signed,
#: log-scale units as Percept.visibility elsewhere. IDENTIFY needs both good
#: light and real proximity to clear; GLINT is set higher so it clears from
#: farther out or dimmer light -- torch vs. glo naturally reach different
#: distances purely because their illuminance falls off differently, with no
#: distance logic of their own.
#:
#: VisualPercept's own falloff is steep (the physical light's 1/r^2 and the
#: percept's own distance term compound to ~1/r^4), so this needs to be tuned
#: against the actual light sources rather than guessed -- see
#: agent_helpers/tune_sword_visibility.py, which walks the real torch/glo
#: lumens through VisualPercept.detectability at a range of distances. 1.5
#: puts a torch's glint right around 3 cells out (detected through r=3, not
#: r=4) and glo's out to ~4-5, matching the intended "brighter light spots it
#: farther away, with no distance logic of its own" behavior. Re-run that
#: script after changing FLAME_FLUX/GLO_LIGHT_COLOR/DIFFUSE_SCALE, since all
#: three shift where this lands.
IDENTIFY_VISIBILITY = -0.5
GLINT_VISIBILITY = 0.6


class Sword(Item):
    """A rusty sword abandoned at a dark dead end, waiting to be found."""

    name = "rusty sword"
    char = u'⸸'   # ⸸ turned dagger
    takeable = True
    mass = 1.2
    length = 90.0
    #: Dull, rust-brown steel. A plain reflector (no emission), so in the dark
    #: it is barely a glint and only reads as a sword once light falls on it.
    fg_color = (0.5, 0.36, 0.26, 1.0)

    #: Seconds before a percept may re-announce itself. The old, far-too-wide
    #: GLINT_VISIBILITY let a slow approach dwell inside the trigger zone for
    #: much longer than a short cooldown, so the same line kept re-firing --
    #: the cooldown itself was working, the zone was just wider than the
    #: window. Now that the zone is corrected this matters less, but a longer
    #: cooldown is cheap insurance against a player who lingers near the edge.
    COOLDOWN = 300.0

    def __init__(self, *args, **kwds):
        Item.__init__(self, *args, **kwds)
        self.percepts = [
            VisualPercept("There is a rusty sword here.", salience=1.0,
                          visibility=IDENTIFY_VISIBILITY, entity=self, cooldown=self.COOLDOWN),
            VisualPercept("Something catches your eye.", salience=1.0,
                          visibility=GLINT_VISIBILITY, entity=self, cooldown=self.COOLDOWN),
        ]
