import numpy as np
from PIL import Image
from .blocktypes import BlockTypes


class Maze:
    """Data defining the landscape.
    """
    def __init__(self, blocks, blocktypes):
        self.blocks = blocks
        self.blocktypes = blocktypes

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


class MazeSprites:
    def __init__(self, maze, txt):
        self._txt = txt
        self._maze = maze
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
