"""Base stats shared by every character: the hero, NPCs, and monsters."""
from .entity import Component


#: Every base stat, in display order, as ``(attribute, abbreviation, label,
#: default)``.
STATS = (
    ('hit_points',      'HP', "hit points",      10),
    ('armor_class',     'AC', "armor class",     10),
    ('weapon_class',    'WC', "weapon class",    10),
    ('stamina',         'SP', "stamina",         10),
    ('magic_power',     'MP', "magic power",     10),
    ('magic_cooldown',  'MC', "magic cooldown",   0),
    ('magic_threshold', 'MT', "magic threshold", 10),
    ('strength',        'ST', "strength",        10),
    ('dexterity',       'DX', "dexterity",       10),
    ('constitution',    'CN', "constitution",    10),
    ('carry_capacity',  'CC', "carry capacity",  10),
)

#: abbreviation -> attribute name, e.g. ``'HP' -> 'hit_points'``
ABBREVIATIONS = {abbr: attr for attr, abbr, _, _ in STATS}


class Stats(Component):
    """A character's base stats, one attribute per entry in :data:`STATS`.

    Stats are a component of the character that owns them, like its inventory
    or location. Each is a plain attribute (``stats.strength``) and can also be
    read or written by abbreviation (``stats['ST']``). Any stat may be
    overridden by keyword at construction; the rest take their default from
    :data:`STATS`.

    Stamina is abbreviated SP (stamina points) so it does not collide with
    strength's ST. Carry capacity is how much weight the character can bear.

    Magic costs nothing to cast, but it is not unlimited: magic cooldown is
    how much magic has been used recently, starting at 0, and magic threshold
    is how much the character can take. Pushing cooldown past the threshold
    risks the character's magic getting stuck on.
    """

    def __init__(self, entity, **overrides):
        Component.__init__(self, entity, component_type='stats')
        for attr, _, _, default in STATS:
            setattr(self, attr, overrides.pop(attr, default))
        if overrides:
            raise TypeError("Unknown stat(s): %s" % ', '.join(sorted(overrides)))

    def __getitem__(self, abbr):
        return getattr(self, ABBREVIATIONS[abbr])

    def __setitem__(self, abbr, value):
        setattr(self, ABBREVIATIONS[abbr], value)

    def as_dict(self):
        """Return ``{abbreviation: value}`` in display order."""
        return {abbr: getattr(self, attr) for attr, abbr, _, _ in STATS}

    def __repr__(self):
        values = ' '.join('%s=%s' % kv for kv in self.as_dict().items())
        return "<Stats %s>" % values
