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


_default_blocktypes = np.array([
    #id  name               char  walkable  opacity fg_color              bg_color              meta
    (0,  'void',            ' ',  False,    0,      (.00, .00, .00, 1.0), (.00, .00, .00, 1.0), {}),
    (1,  'path',            '.',  True,     0,      (.20, .20, .20, 1.0), (.05, .05, .05, 1.0), {}),
    (2,  'wall',            '#',  False,    1,      (.00, .00, .00, 1.0), (.20, .20, .20, 1.0), {'bg_color_var': 0.01}),
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
        return self.data['char'].tostring().decode()
