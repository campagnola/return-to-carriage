"""Game-owned render layers: the bridge between game state and renderers.

The game writes what should be displayed into these layers (dense or sparse
sprite collections and float scalar/color fields); rendering backends read
them and draw by whatever means they like. Layers are pure numpy — no
rendering library is imported here.

Change tracking is by integer version counters, not events: a write costs one
numpy slice assignment plus an increment. Backends compare counters once per
frame and re-upload only what changed.

Each layer also has a single-slot ``observer`` callback (default None) invoked
whenever its version is bumped. This exists so an interactive backend can
learn "something changed, schedule a frame" without polling; it carries no
information (backends still diff version counters at sync time) and must stay
cheap and re-entrancy-safe (e.g. vispy's canvas update(), which coalesces).

- ``version`` is bumped on any data write.
- ``structure_version`` is additionally bumped when the underlying arrays are
  reallocated (new sprites added or a slot reshaped), meaning any references a
  backend holds into the old arrays are stale.

Conventions:
- Sprite positions are float32 (x, y, z); a NaN position hides the sprite.
- ``glyph`` values are ids from a GlyphRegistry; backends map ids to their own
  representation (texture atlas index, terminal character, ...).
"""

import numpy as np


class GlyphRegistry(object):
    """Append-only registry mapping characters to small integer glyph ids.

    Ids are assigned in insertion order. Mirrors the CharAtlas.add_chars
    contract (add_chars returns the first new id and appends unconditionally)
    so that a backend feeding ``chars`` to its own atlas in order gets an
    identity id mapping.
    """
    def __init__(self):
        self._char_ids = {}
        self.chars = []
        self.version = 0
        self.observer = None

    def __len__(self):
        return len(self.chars)

    def __getitem__(self, char):
        """Return the id for *char*, adding it if not present."""
        if char not in self._char_ids:
            return self.add_chars(char)
        return self._char_ids[char]

    def add_chars(self, chars):
        """Add characters (a string or sequence of 1-char strings); return the first new id."""
        first = len(self.chars)
        for i, char in enumerate(chars):
            self._char_ids[char] = first + i
            self.chars.append(char)
        self.version += 1
        if self.observer is not None:
            self.observer()
        return first


class SpriteLayer(object):
    """A collection of positioned character sprites owned by the game.

    Regions are allocated with add_sprites(), which returns a SpriteSlot
    handle used to write position/glyph/color data. All slots share one set
    of contiguous arrays so a backend can upload the layer in one call.
    """
    def __init__(self, name=None):
        self.name = name
        self.position = np.empty((0, 3), dtype='float32')
        self.glyph = np.empty((0,), dtype='uint32')
        self.fgcolor = np.empty((0, 4), dtype='float32')
        self.bgcolor = np.empty((0, 4), dtype='float32')
        self.slots = []
        self.version = 0
        self.structure_version = 0
        self.observer = None

    def __len__(self):
        return self.position.shape[0]

    def add_sprites(self, shape):
        """Allocate a region of sprites, return a SpriteSlot of the given shape.

        New sprites start hidden (NaN position).
        """
        if not isinstance(shape, tuple):
            raise TypeError("shape must be a tuple (got %r)" % (shape,))
        n = int(np.prod(shape))
        old_size = self._resize(len(self) + n)
        slot = SpriteSlot(self, start=old_size, shape=shape)
        self.slots.append(slot)
        return slot

    def _resize(self, n):
        """Resize the shared arrays to n sprites, return the old size.

        Data in the common prefix is preserved; rows beyond it start hidden.
        """
        n1 = len(self)
        keep = min(n, n1)

        position = np.full((n, 3), np.nan, dtype='float32')
        glyph = np.zeros((n,), dtype='uint32')
        fgcolor = np.zeros((n, 4), dtype='float32')
        bgcolor = np.zeros((n, 4), dtype='float32')
        position[:keep] = self.position[:keep]
        glyph[:keep] = self.glyph[:keep]
        fgcolor[:keep] = self.fgcolor[:keep]
        bgcolor[:keep] = self.bgcolor[:keep]
        self.position, self.glyph, self.fgcolor, self.bgcolor = position, glyph, fgcolor, bgcolor

        self.version += 1
        self.structure_version += 1
        if self.observer is not None:
            self.observer()
        return n1

    def _slot_shape_changed(self):
        """Repack slot regions after a slot changed shape (cf. SpritesVisual.data_changed_shape)."""
        size = sum(len(slot) for slot in self.slots)
        self._resize(size)
        start = 0
        for slot in self.slots:
            slot.set_start(start)
            start += len(slot)

    def _data_changed(self):
        self.version += 1
        if self.observer is not None:
            self.observer()


class SpriteSlot(object):
    """Handle for writing to a contiguous region of sprites in a SpriteLayer.

    Mirrors the SpriteData API: property setters accept anything numpy can
    broadcast to the slot's shape (scalars, tuples, arrays; None writes NaN).
    Assigned values are cached so they can be re-applied when the layer
    repacks its arrays.
    """
    def __init__(self, layer, start, shape):
        self.layer = layer
        self.set_shape(shape, inform_parent=False)
        self.set_start(start)

    def __len__(self):
        return int(np.prod(self.shape))

    @property
    def position(self):
        start, stop = self.indices
        return self.layer.position[start:stop].reshape(self.shape + (3,))

    @position.setter
    def position(self, p):
        self._position = p
        self.position[:] = p
        self.layer._data_changed()

    @property
    def glyph(self):
        start, stop = self.indices
        return self.layer.glyph[start:stop].reshape(self.shape)

    @glyph.setter
    def glyph(self, p):
        self._glyph = p
        self.glyph[:] = p
        self.layer._data_changed()

    @property
    def fgcolor(self):
        start, stop = self.indices
        return self.layer.fgcolor[start:stop].reshape(self.shape + (4,))

    @fgcolor.setter
    def fgcolor(self, p):
        self._fgcolor = p
        self.fgcolor[:] = p
        self.layer._data_changed()

    @property
    def bgcolor(self):
        start, stop = self.indices
        return self.layer.bgcolor[start:stop].reshape(self.shape + (4,))

    @bgcolor.setter
    def bgcolor(self, p):
        self._bgcolor = p
        self.bgcolor[:] = p
        self.layer._data_changed()

    def set_start(self, start):
        self.indices = (start, start + len(self))
        if self._position is not None:
            self.position = self._position
            self.glyph = self._glyph
            self.fgcolor = self._fgcolor
            self.bgcolor = self._bgcolor

    def set_shape(self, shape, inform_parent=True):
        self.shape = shape
        self._position = None
        self._glyph = None
        self._fgcolor = None
        self._bgcolor = None
        if inform_parent:
            self.layer._slot_shape_changed()


class FieldLayer(object):
    """A named float32 scalar/vector field covering the maze (light, LOS, memory, ...)."""
    def __init__(self, name, shape=None, data=None):
        self.name = name
        if data is not None:
            self.data = np.ascontiguousarray(data, dtype='float32')
        else:
            self.data = np.zeros(shape, dtype='float32')
        self.version = 0
        self.observer = None

    def set_data(self, data):
        """Copy *data* into the field (in place when shapes match) and bump version."""
        data = np.asarray(data)
        if data.shape == self.data.shape:
            self.data[...] = data
        else:
            self.data = np.ascontiguousarray(data, dtype='float32')
        self.version += 1
        if self.observer is not None:
            self.observer()

    def bump(self):
        """Declare that self.data was mutated in place."""
        self.version += 1
        if self.observer is not None:
            self.observer()
