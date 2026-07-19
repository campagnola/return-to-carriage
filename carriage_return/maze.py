import numpy as np
from PIL import Image
from .blocktypes import BlockTypes
from .geometry import isocurve
from .entity import Entity
from .inventory import Inventory
from .location import Location


class Maze(Entity):
    """Data defining the landscape.
    """
    def __init__(self, blocks, blocktypes, obj_name=None):
        Entity.__init__(self, entity_type='maze', obj_name=obj_name)
        self.blocks = blocks
        self.blocktypes = blocktypes

        # all objects in maze by location
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

    def invalidate_appearance(self):
        """Drop the cached opacity/colour arrays after ``blocks`` was edited.

        The three are derived from ``blocks`` and cached on first use, so
        anything that writes into ``blocks`` after construction -- stamping a
        portal end, most notably -- must call this or the maze keeps drawing
        (and casting shadows) as it looked beforehand.
        """
        self._opacity = None
        self._fg_color = None
        self._bg_color = None

    @classmethod
    def filled(cls, shape, blocktypes, blocktype='wall', obj_name=None):
        """A maze of *shape* ``(rows, cols)`` filled with one block type."""
        blocks = np.full(shape, blocktypes.id_of(blocktype), dtype='uint8')
        return cls(blocks, blocktypes, obj_name=obj_name)

    def blocktype_at(self, i, j):
        bid = self.blocks[i, j]
        return self.blocktypes[bid]

    @property
    def opacity(self):
        if self._opacity is None:
            self._opacity = self.blocktypes['opacity'][self.blocks].astype('float32')
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
    def load_image(cls, filename, blocktypes=None, obj_name=None):
        """Build a maze from an image: non-black pixels are wall, black is path.

        *blocktypes* defaults to a fresh table, but a multi-level world passes
        its shared one so that block ids mean the same thing on every level.
        """
        if blocktypes is None:
            blocktypes = BlockTypes()
        maze_blocks = np.array(Image.open(filename))[::-1,:,0]
        maze_blocks[maze_blocks>0] = blocktypes.id_of('wall')
        maze_blocks[maze_blocks==0] = blocktypes.id_of('path')
        return cls(maze_blocks, blocktypes, obj_name=obj_name)

    def add_scenery(self, glyphs, layer):
        """Fill a scenery SpriteLayer with this maze's blocks; return the slot.

        *glyphs* is the scene's GlyphRegistry, *layer* the scenery SpriteLayer.
        The caller owns the returned SpriteSlot and frees it (via
        layer.remove_sprites/clear) when this maze stops being displayed --
        a maze may be built into more than one layer over its lifetime, so it
        does not keep a reference of its own.
        """
        # all_chars is in blocktype-id order, so indexing the returned char->id
        # mapping in that order gives a blocktype id -> glyph id lookup table.
        # A table rather than a scalar offset because the registry deduplicates:
        # glyph ids for these chars are not necessarily contiguous, and two
        # blocktypes sharing a char correctly map to the same glyph.
        chars = self.blocktypes.all_chars
        char_ids = glyphs.add_chars(chars)
        glyph_ids = np.array([char_ids[c] for c in chars], dtype='uint32')

        scenery = layer.add_sprites(self.shape)
        scenery.glyph = glyph_ids[self.blocks]

        # set positions
        shape = self.shape
        pos = np.zeros(shape + (3,), dtype='float32')
        pos[..., :2] = np.mgrid[0:shape[1], 0:shape[0]].transpose(2, 1, 0)
        scenery.position = pos

        # set colors
        scenery.fgcolor = self.fg_color
        scenery.bgcolor = self.bg_color

        return scenery

    def opaque_geometry(self):
        """Return a list of vertex loops defining the boundaries of objects that block line-of-sight.
        """
        m = self._opaque_geometry_mask()
        return isocurve(m.astype(float), level=0.5, connected=True)

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
