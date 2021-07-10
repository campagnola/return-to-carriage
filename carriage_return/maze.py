import numpy as np
import vispy
from PIL import Image
from .blocktypes import BlockTypes
from .inventory import Inventory
from .location import Location
from .entity_type import EntityType


class Maze:
    """Data defining the landscape.
    """
    def __init__(self, blocks, blocktypes):
        self.blocks = blocks
        self.blocktypes = blocktypes

        # all objects in maze by location
        self.type = EntityType('maze')
        self.inventory = Inventory(
            entity=self, 
            slot_type=tuple,
            allowed_slots=[(j, i) for j in range(blocks.shape[1]) for i in range(blocks.shape[0])]
        )
        self.location = Location(self, None, None)

        self._opacity = None
        self._fg_color = None
        self._bg_color = None

    @property
    def shape(self):
        return self.blocks.shape

    def blocktype_at(self, i, j):
        bid = self.blocks[i, j]
        return self.blocktypes[bid]

    @property
    def opacity(self):
        if self._opacity is None:
            self._opacity = self.blocktypes['opacity'][self.blocks]
        return self._opacity

    @property
    def fg_color(self):
        if self._fg_color is None:
            self._fg_color = self.blocktypes['fg_color'][self.blocks]
        return self._fg_color

    @property
    def bg_color(self):
        if self._bg_color is None:
            self._bg_color = self.blocktypes['bg_color'][self.blocks]

            # randomize colors
            for bt in self.blocktypes:
                if 'bg_color_var' in bt['meta']:
                    mask = self.blocks == bt['id']
                    rand = np.random.normal(scale=bt['meta']['bg_color_var'], size=(mask.sum(), 1))
                    self._bg_color[mask] += rand

        return self._bg_color

    @classmethod
    def load_image(cls, filename):
        blocktypes = BlockTypes()
        maze_blocks = np.array(Image.open(filename))[::-1,:,0]
        maze_blocks[maze_blocks>0] = blocktypes.id_of('wall')
        maze_blocks[maze_blocks==0] = blocktypes.id_of('path')
        return cls(maze_blocks, blocktypes)

    def add_sprites(self, char_atlas, txt):
        self.sprites = MazeSprites(self, char_atlas, txt)

    def opaque_geometry(self):
        """Return a list of vertex loops defining the boundaries of objects that block line-of-sight.
        """
        m = self._opaque_geometry_mask()
        return vispy.geometry.isocurve.isocurve(m.astype(float), level=0.5, connected=True)

    def _opaque_geometry_mask(self):
        opaque = self.opacity > 0.5
        padded = np.zeros((opaque.shape[0] + 2, opaque.shape[1] + 2), dtype=bool)
        padded[1:-1, 1:-1] = opaque
        opaque_mask = np.empty((opaque.shape[0] * 3, opaque.shape[1] * 3), dtype=bool)

        opaque_mask[1::3, 1::3] = opaque

        opaque_mask[0::3, 1::3] = padded[:-2,  1:-1] & opaque
        opaque_mask[2::3, 1::3] = padded[2:,   1:-1] & opaque
        opaque_mask[1::3, 0::3] = padded[1:-1,  :-2] & opaque
        opaque_mask[1::3, 2::3] = padded[1:-1,   2:] & opaque

        opaque_mask[0::3, 0::3] = padded[:-2, :-2] & opaque_mask[0::3, 1::3] & opaque_mask[1::3, 0::3]
        opaque_mask[2::3, 0::3] = padded[2:,  :-2] & opaque_mask[2::3, 1::3] & opaque_mask[1::3, 0::3]
        opaque_mask[0::3, 2::3] = padded[:-2,  2:] & opaque_mask[0::3, 1::3] & opaque_mask[1::3, 2::3]
        opaque_mask[2::3, 2::3] = padded[2:,   2:] & opaque_mask[2::3, 1::3] & opaque_mask[1::3, 2::3]

        return opaque_mask
        


class MazeSprites:
    def __init__(self, maze, char_atlas, txt):
        self._txt = txt
        self._maze = maze
        self._char_atlas = char_atlas
        char_atlas.add_chars(maze.blocktypes.all_chars)

        self.sprites = txt.add_sprites(maze.shape)
        self.sprites.sprite = maze.blocks

        # set positions
        shape = maze.shape
        pos = np.zeros(shape + (3,), dtype='float32')
        pos[...,:2] = np.mgrid[0:shape[1], 0:shape[0]].transpose(2, 1, 0)
        self.sprites.position = pos

        # set colors
        self.sprites.fgcolor = maze.fg_color
        self.sprites.bgcolor = maze.bg_color
