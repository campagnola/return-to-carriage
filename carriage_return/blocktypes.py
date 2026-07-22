import numpy as np


blocktype_dtype = [
    ('id', int), 
    ('name', object), 
    ('char', 'S1'),
    ('walkable', bool), 
    ('opacity', float),
    ('fg_color', 'float32', 4),
    ('bg_color', 'float32', 4),
    ('meta', object),
]


# 'path' and 'wall' must keep ids 1 and 2: Maze.load_image thresholds the image
# against them by id. Everything else is appended, which BlockTypes.add is built
# for -- it renumbers in order, so id == index continues to hold.
_default_blocktypes = np.array([
    #id  name               char  walkable  opacity fg_color              bg_color              meta
    (0,  'void',            ' ',  False,    0,      (.00, .00, .00, 1.0), (.00, .00, .00, 1.0), {}),
    (1,  'path',            '.',  True,     0,      (.20, .20, .20, 1.0), (.10, .10, .10, 1.0), {'bg_color_var': 0.005}),
    (2,  'wall',            '#',  False,    1,      (.00, .00, .00, 1.0), (.40, .40, .40, 1.0), {'bg_color_var': 0.03}),
    # A barred window to the outside: impassable, but not opaque -- you see the
    # bright sky through the bars, and daylight streams past them. Dark bars
    # (fg) silhouette against a bright cool-sky background.
    (3,  'grate',           '#',  False,    0,      (.05, .05, .07, 1.0), (.55, .70, 1.0, 1.0), {}),
], dtype=blocktype_dtype)


class BlockTypes:
    def __init__(self, bt_array=None):
        self.data = (bt_array or _default_blocktypes).copy()
        assert np.all(self.data['id'] == np.arange(len(self.data)))
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

        *new_blocktypes* must be an array with dtype=blocktype_dtype
        """
        self.data = np.concatenate([self.data, new_blocktypes])
        self.data['id'] = np.arange(len(self))
        self._update()

    def _update(self):    
        self.by_name = {bt['name']:bt for bt in self.data}

    @property
    def all_chars(self):
        return self.data['char'].tobytes().decode()
