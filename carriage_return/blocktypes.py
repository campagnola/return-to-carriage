import numpy as np


blocktype_dtype = [
    ('id', int),
    ('name', object),
    ('char', 'S1'),
    ('walkable', bool),
    ('opacity', float),
    ('fg_color', 'float32', 4),
    ('bg_color', 'float32', 4),
    ('bg_color_var', float),
    ('fg_emission', 'float32', 3),
    ('bg_emission', 'float32', 3),
]


def blocktype(name, char, walkable, opacity, fg_color, bg_color, bg_color_var=0.0,
              fg_emission=(0, 0, 0), bg_emission=(0, 0, 0)):
    """One row of the blocktype table, as a list of columns in dtype order.

    Ids are not given here -- BlockTypes assigns them by position. Code that
    depends on a stable encoding (e.g. Maze.load_image) maps its own values to
    blocktype *names*, never to ids.

    New per-blocktype attributes get a new keyword argument here (with a
    default) plus a field in *blocktype_dtype* -- not a dict dumped into a
    catch-all column.
    """
    return [name, char, walkable, opacity, fg_color, bg_color, bg_color_var, fg_emission, bg_emission]


_default_blocktypes = [
    #         name     char  walkable  opacity  fg_color              bg_color
    blocktype('void',  ' ',  False,    0,       (.00, .00, .00, 1.0), (.00, .00, .00, 1.0)),
    blocktype('path',  '.',  True,     0,       (.20, .20, .20, 1.0), (.10, .10, .10, 1.0), bg_color_var=0.005),
    blocktype('wall',  '#',  False,    1,       (.00, .00, .00, 1.0), (.40, .40, .40, 1.0), bg_color_var=0.03),
    # A barred window to the outside: impassable, but not opaque -- you see the
    # bright sky through the bars, and daylight streams past them. Dark bars
    # (fg) silhouette against a bright cool-sky background.
    blocktype('grate', '#',  False,    0,       (.05, .05, .07, 1.0), (.55, .70, 1.0, 1.0),
              fg_emission=(0.05, 0.03, 0.0), bg_emission=(0.05, 0.03, 0.0)),
    blocktype('grass', '.',  True,     0,       (0.01, 0.16, 0.02, 1.0),    (0.01, 0.15, 0.01, 1.0), bg_color_var=0.003),
    # A river: impassable like a wall (you cannot wade it, only cross by a
    # bridge) but not opaque, so it neither blocks sight nor casts a shadow.
    blocktype('river',  '~',  False,   0,       (.10, .30, .50, 1.0), (.05, .15, .45, 1.0), bg_color_var=0.002),
    blocktype('dirt',   ':',  True,    0,       (.35, .25, .15, 1.0), (.30, .20, .10, 1.0), bg_color_var=0.01),
    blocktype('bridge', '=',  True,    0,       (.45, .32, .18, 1.0), (.15, .12, .06, 1.0), bg_color_var=0.02),
    # A river's dry bank: loose sand/gravel just outside the water's edge.
    # Warm and light -- brighter than water.RIVER_BED_ALBEDO's submerged,
    # silty tone, since this sand is never underwater.
    blocktype('sand',   ',',  True,    0,       (.78, .70, .50, 1.0), (.72, .63, .42, 1.0), bg_color_var=0.006),
]


def _to_recarray(rows):
    """Build the blocktype recarray from blocktype() rows, assigning ids by position."""
    data = np.array([(0, *row) for row in rows], dtype=blocktype_dtype)
    data['id'] = np.arange(len(data))
    return data


class BlockTypes:
    def __init__(self, rows=None):
        self.data = _to_recarray(rows if rows is not None else _default_blocktypes)
        self._update()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def get(self, name):
        return self.by_name[name]

    def id_of(self, name):
        return self.by_name[name]['id']

    def add(self, new_blocktypes):
        """Add new blocktypes to this instance.

        *new_blocktypes* is a list of blocktype() rows.
        """
        self.data = np.concatenate([self.data, _to_recarray(new_blocktypes)])
        self.data['id'] = np.arange(len(self))
        self._update()

    def _update(self):
        self.by_name = {bt['name']:bt for bt in self.data}

    @property
    def all_chars(self):
        return self.data['char'].tobytes().decode()
