import numpy as np
from collections import defaultdict
from PIL import Image
from .blocktypes import BlockTypes
from .events import Observable
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

        # the Level this maze belongs to, set by Level.__init__. The back
        # reference is what lets an entity get from its location to the sight
        # fields of the level it is actually standing on, rather than to
        # whichever level the scene happens to be displaying.
        self.level = None

        self._opacity = None
        self._fg_color = None
        self._bg_color = None
        self._fg_emission = None
        self._bg_emission = None

        # per-cell modifier registry: survives level rebuilds
        self._cell_modifiers = defaultdict(list)  # (row, col) → [ColorModifier, ...]
        self._scenery_slot = None  # the SpriteSlot currently displaying this maze
        self.appearance_changed = Observable()

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
        self._fg_emission = None
        self._bg_emission = None

    def add_cell_modifier(self, pos, modifier):
        """Register a ColorModifier for maze cell pos=(x, y).

        If a scenery slot currently exists (i.e. the maze is displayed), attach
        the modifier immediately. Otherwise it will be attached when add_scenery
        is next called.
        """
        x, y = int(pos[0]), int(pos[1])
        self._cell_modifiers[(x, y)].append(modifier)
        if self._scenery_slot is not None:
            modifier.cells = y * self.shape[1] + x
            self._scenery_slot.add_modifier(modifier)
        self.appearance_changed()

    def remove_cell_modifier(self, pos, modifier):
        """Unregister a ColorModifier for maze cell pos=(x, y)."""
        x, y = int(pos[0]), int(pos[1])
        mods = self._cell_modifiers.get((x, y), [])
        if modifier in mods:
            mods.remove(modifier)
        if self._scenery_slot is not None:
            try:
                self._scenery_slot.remove_modifier(modifier)
            except ValueError:
                pass  # modifier not attached (slot rebuilt since it was added)
        self.appearance_changed()

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
                if bt['bg_color_var'] > 0:
                    mask = self.blocks == bt['id']
                    rand = np.random.normal(scale=bt['bg_color_var'], size=(mask.sum(), 1))
                    self._bg_color[mask] += rand

        return self._bg_color

    @property
    def fg_emission(self):
        if self._fg_emission is None:
            self._fg_emission = self.blocktypes['fg_emission'][self.blocks]
        return self._fg_emission

    @property
    def bg_emission(self):
        if self._bg_emission is None:
            self._bg_emission = self.blocktypes['bg_emission'][self.blocks]
        return self._bg_emission

    @classmethod
    def load_image(cls, filename, blocktypes=None, encoding=None, obj_name=None):
        """Build a maze from an image whose pixel values encode blocktypes.

        *encoding* maps a pixel value to a blocktype name; it describes how the
        image was drawn. Block ids are assigned automatically and need not match
        pixel values, so the image's encoding is stated by name. It defaults to
        a binary scheme: black is path, white is wall.

        *blocktypes* defaults to a fresh table, but a multi-level world passes
        its shared one so that block ids mean the same thing on every level.
        """
        if blocktypes is None:
            blocktypes = BlockTypes()
        if encoding is None:
            encoding = {0: 'path', 255: 'wall'}
        pixels = np.array(Image.open(filename))[::-1,:,0]
        unknown = set(np.unique(pixels)) - set(encoding)
        assert not unknown, f"{filename} has pixel values {sorted(unknown)} not in encoding"
        blocks = np.zeros(pixels.shape, dtype='uint8')
        for value, name in encoding.items():
            blocks[pixels == value] = blocktypes.id_of(name)
        return cls(blocks, blocktypes, obj_name=obj_name)

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

        # set colors and base emission
        scenery.fgcolor = self.fg_color
        scenery.bgcolor = self.bg_color
        scenery.fg_emission = self.fg_emission.reshape(-1, 3)
        scenery.bg_emission = self.bg_emission.reshape(-1, 3)

        # store slot so add_cell_modifier can attach to it immediately, and
        # reattach any modifiers that were registered before this call
        self._scenery_slot = scenery
        cols = self.shape[1]
        for (x, y), mods in self._cell_modifiers.items():
            idx = y * cols + x
            for mod in mods:
                mod.cells = idx
                scenery.add_modifier(mod)

        return scenery

    def add_light(self, light, pos):
        """Attach *light* to this maze at cell *pos* ``(x, y)``.

        For light that belongs to the map itself rather than to anything that
        moves -- a shaft of daylight through a hole in the ceiling, say. The
        light shines from *pos* for as long as this maze's level is shown,
        whatever walks beneath it. *light* is a :class:`~.light.Light` whose
        host entity is this maze; returns it for convenience.
        """
        light.pin(self, pos)
        return light

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
