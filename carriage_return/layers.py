"""Game-owned render layers: the bridge between game state and renderers.

The game writes what should be displayed into these layers (dense or sparse
sprite collections and float scalar/color fields); rendering backends read
them and draw by whatever means they like. Layers are pure numpy — no
rendering library is imported here.

Change tracking is by integer version counters, not events: a write costs one
numpy slice assignment plus an increment. Backends compare counters once per
frame and re-upload only what changed.

Each layer also has a ``changed`` Observable (see events.py) invoked with no
arguments whenever its version is bumped. This exists so an interactive
backend can learn "something changed, schedule a frame" without polling; it
carries no information (backends still diff version counters at sync time)
and subscribers must stay cheap and re-entrancy-safe (e.g. vispy's canvas
update(), which coalesces).

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

from .events import Observable
from .glyph_effects import ColorModifier  # noqa: F401 — re-exported for callers


def _offset_cells(cells, offset):
    """Shift a cell index or slice by *offset* (a slot's start position in the layer)."""
    if isinstance(cells, slice):
        start = (cells.start or 0) + offset
        stop = (cells.stop + offset) if cells.stop is not None else None
        return slice(start, stop, cells.step)
    return cells + offset  # integer index


class GlyphLayer(object):
    """Base for game-owned glyph layers: the shared change-tracking contract.

    Every layer kind (dense CharGridLayer, sparse SpriteLayer) carries glyph
    ids from a GlyphRegistry plus float32 fg/bg colors, and publishes changes
    through the same three attributes:

    - ``version`` bumps on any data write.
    - ``structure_version`` additionally bumps when the underlying arrays are
      reallocated, meaning references a backend holds into them are stale.
    - ``changed`` is an Observable invoked (with no arguments) after any
      version bump; subscribers must stay cheap and thread-safe (e.g. set a
      dirty flag) — backends still diff version counters at sync time.
    """
    def __init__(self, name=None):
        self.name = name
        self.version = 0
        self.structure_version = 0
        self.changed = Observable()

    def _changed(self, structure=False):
        self.version += 1
        if structure:
            self.structure_version += 1
        self.changed()


class GlyphRegistry(object):
    """Append-only registry mapping characters to small integer glyph ids.

    Ids are assigned in insertion order and never reused; a char already
    present keeps its id, so ``chars[id] == char`` always holds. A backend
    that feeds each batch of *newly appended* chars (``chars[n_synced:]``) to
    its own atlas in order therefore gets an identity id -> atlas index
    mapping.
    """
    def __init__(self):
        self._char_ids = {}
        self.chars = []
        self.version = 0
        self.changed = Observable()

    def __len__(self):
        return len(self.chars)

    def __getitem__(self, char):
        """Return the id for *char*, adding it if not present."""
        if char not in self._char_ids:
            self.add_chars(char)
        return self._char_ids[char]

    def add_chars(self, chars):
        """Add characters (a string or sequence of 1-char strings); return a dict mapping each char to its glyph id."""
        next_id = len(self.chars)
        lut = {}
        for i, char in enumerate(chars):
            if char in self._char_ids:
                lut[char] = self._char_ids[char]
                continue
            self._char_ids[char] = next_id
            lut[char] = next_id
            self.chars.append(char)
            next_id += 1
        self.version += 1
        self.changed()
        return lut


class SpriteLayer(GlyphLayer):
    """A collection of positioned character sprites owned by the game.

    The sparse layer kind: every sprite has its own free (x, y, z) position.
    Regions are allocated with add_sprites(), which returns a SpriteSlot
    handle used to write position/glyph/color data. All slots share one set
    of contiguous arrays so a backend can upload the layer in one call.
    """
    def __init__(self, name=None):
        GlyphLayer.__init__(self, name=name)
        self.position = np.empty((0, 3), dtype='float32')
        self.glyph = np.empty((0,), dtype='uint32')
        self.fgcolor = np.empty((0, 4), dtype='float32')
        self.bgcolor = np.empty((0, 4), dtype='float32')
        # Linear emitted radiance per sprite (RGB); 0 for a plain reflective
        # glyph, non-zero for an emitter such as a torch flame.
        self.fg_emission = np.empty((0, 3), dtype='float32')
        self.bg_emission = np.empty((0, 3), dtype='float32')
        self.slots = []

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

    def remove_sprites(self, slot):
        """Free *slot*, repacking the remaining slots into contiguous arrays.

        Bumps ``structure_version``, so a backend holding references into the
        old arrays re-reads them at its next sync. The freed slot must not be
        written to afterwards.
        """
        self.slots.remove(slot)
        slot.layer = None
        self._slot_shape_changed()

    def clear(self):
        """Free every slot, leaving the layer empty.

        Used when the world content a layer draws is replaced wholesale --
        switching levels rebuilds the scenery layer from the new maze.
        """
        for slot in self.slots:
            slot.layer = None
        self.slots = []
        self._slot_shape_changed()

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
        fg_emission = np.zeros((n, 3), dtype='float32')
        bg_emission = np.zeros((n, 3), dtype='float32')
        position[:keep] = self.position[:keep]
        glyph[:keep] = self.glyph[:keep]
        fgcolor[:keep] = self.fgcolor[:keep]
        bgcolor[:keep] = self.bgcolor[:keep]
        fg_emission[:keep] = self.fg_emission[:keep]
        bg_emission[:keep] = self.bg_emission[:keep]
        self.position, self.glyph, self.fgcolor, self.bgcolor = position, glyph, fgcolor, bgcolor
        self.fg_emission = fg_emission
        self.bg_emission = bg_emission

        self._changed(structure=True)
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
        self._changed()


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
        self._modifiers = []
        self._dirty = False
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
        if self._modifiers:
            self._dirty = True
            self._recompose()
        else:
            self.fgcolor[:] = p
            self.layer._data_changed()

    @property
    def bgcolor(self):
        start, stop = self.indices
        return self.layer.bgcolor[start:stop].reshape(self.shape + (4,))

    @bgcolor.setter
    def bgcolor(self, p):
        self._bgcolor = p
        if self._modifiers:
            self._dirty = True
            self._recompose()
        else:
            self.bgcolor[:] = p
            self.layer._data_changed()

    @property
    def fg_emission(self):
        start, stop = self.indices
        return self.layer.fg_emission[start:stop].reshape(self.shape + (3,))

    @fg_emission.setter
    def fg_emission(self, p):
        self._fg_emission = p
        if self._modifiers:
            self._dirty = True
            self._recompose()
        else:
            start, stop = self.indices
            self.layer.fg_emission[start:stop] = p
            self.layer._data_changed()

    @property
    def bg_emission(self):
        start, stop = self.indices
        return self.layer.bg_emission[start:stop].reshape(self.shape + (3,))

    @bg_emission.setter
    def bg_emission(self, p):
        self._bg_emission = p
        if self._modifiers:
            self._dirty = True
            self._recompose()
        else:
            start, stop = self.indices
            self.layer.bg_emission[start:stop] = p
            self.layer._data_changed()

    def set_start(self, start):
        self.indices = (start, start + len(self))
        if self._position is not None:
            self.position = self._position
            self.glyph = self._glyph
            self.fgcolor = self._fgcolor
            self.bgcolor = self._bgcolor
            # optional: only slots that emit ever set these; others keep the 0
            # the layer's arrays default to after a repack
            if self._fg_emission is not None:
                self.fg_emission = self._fg_emission
            if self._bg_emission is not None:
                self.bg_emission = self._bg_emission
            if self._modifiers:
                # The individual setters above called _recompose when
                # _modifiers was non-empty; call once more here to cover
                # the case where only position/glyph changed (no setter
                # triggered _recompose) so the modifier deltas land at
                # the new indices.
                self._recompose()

    def set_shape(self, shape, inform_parent=True):
        self.shape = shape
        self._position = None
        self._glyph = None
        self._fgcolor = None
        self._bgcolor = None
        self._fg_emission = None
        self._bg_emission = None
        if inform_parent:
            self.layer._slot_shape_changed()

    def add_modifier(self, modifier):
        """Attach a ColorModifier; mark dirty so the renderer recomposes."""
        modifier._slot = self
        self._modifiers.append(modifier)
        self._dirty = True
        if self.layer is not None:
            self.layer._data_changed()

    def remove_modifier(self, modifier):
        """Detach a ColorModifier; mark dirty so the renderer restores base."""
        self._modifiers.remove(modifier)
        modifier._slot = None
        self._dirty = True
        if self.layer is not None:
            self.layer._data_changed()

    def _recompose(self):
        """Write effective = base + Σ modifiers into the layer arrays.

        Called on the draw thread (via VispyLayerRenderer.sync) and eagerly
        from setters when modifiers are present. Resets the dirty flag.
        """
        self._dirty = False
        start, stop = self.indices
        n = stop - start
        layer = self.layer

        def _flat(val, ncols):
            """Return *val* reshaped to (n, ncols) when its total size matches;
            otherwise return as-is so numpy can broadcast scalar-like values."""
            a = np.asarray(val)
            return a.reshape(n, ncols) if a.size == n * ncols else a

        # Re-apply base values; reset to 0 if base is unset but modifiers
        # will add to it, so stale accumulated values don't leak through.
        if self._fg_emission is not None:
            layer.fg_emission[start:stop] = _flat(self._fg_emission, 3)
        elif self._modifiers:
            layer.fg_emission[start:stop] = 0
        if self._bg_emission is not None:
            layer.bg_emission[start:stop] = _flat(self._bg_emission, 3)
        elif self._modifiers:
            layer.bg_emission[start:stop] = 0
        if self._fgcolor is not None:
            layer.fgcolor[start:stop] = _flat(self._fgcolor, 4)
        elif self._modifiers:
            layer.fgcolor[start:stop] = 0
        if self._bgcolor is not None:
            layer.bgcolor[start:stop] = _flat(self._bgcolor, 4)
        elif self._modifiers:
            layer.bgcolor[start:stop] = 0

        for mod in self._modifiers:
            cells = (slice(start, stop) if mod.cells == slice(None)
                     else _offset_cells(mod.cells, start))
            if mod.fg_emission is not None:
                layer.fg_emission[cells] += mod.fg_emission
            if mod.bg_emission is not None:
                layer.bg_emission[cells] += mod.bg_emission
            if mod.fgcolor is not None:
                layer.fgcolor[cells] += mod.fgcolor
            if mod.bgcolor is not None:
                layer.bgcolor[cells] += mod.bgcolor

        layer._data_changed()


class CharGridLayer(GlyphLayer):
    """A dense rows×cols block of character cells owned by the game.

    The dense layer kind: menus, pagers, console and HUD text. Where the
    grid draws is an attribute, not part of the concept — ``space`` is
    'screen' (anchored to the canvas; the only space in use) or 'world'
    (maze coordinates; reserved). ``anchor`` names a canvas position:
    'center', 'top', 'bottom', 'left', 'right', or a corner ('top-left',
    'top-right', 'bottom-left', 'bottom-right'). ``offset`` is a (rows,
    cols) displacement in cells from the anchored edge toward the canvas
    interior (applied as down/right for 'center').

    Cell contents are glyph ids from the given GlyphRegistry. Grids have
    fixed shapes; renderers reposition them on resize but never reshape
    them. All meaning (borders, cursors, titles) is written into the cells
    by game-side painters — renderers draw a rectangle of cells and nothing
    else.
    """
    def __init__(self, registry, shape, space='screen', anchor='center',
                 offset=(0, 0), name=None):
        GlyphLayer.__init__(self, name=name)
        self.registry = registry
        self.shape = tuple(shape)
        self.space = space
        self.anchor = anchor
        self.offset = tuple(offset)
        self.glyph = np.full(self.shape, registry[' '], dtype='uint32')
        self.fgcolor = np.ones(self.shape + (4,), dtype='float32')
        self.bgcolor = np.zeros(self.shape + (4,), dtype='float32')

    def reshape(self, shape):
        """Reallocate to *shape* (rows, cols), resetting every cell to blank.

        Bumps ``structure_version`` — backend references into the old arrays
        are stale. Callers repaint the grid immediately afterwards.
        """
        self.shape = tuple(shape)
        self.glyph = np.full(self.shape, self.registry[' '], dtype='uint32')
        self.fgcolor = np.ones(self.shape + (4,), dtype='float32')
        self.bgcolor = np.zeros(self.shape + (4,), dtype='float32')
        self._changed(structure=True)

    def set_data(self, chars, fgcolor, bgcolor):
        """Replace the grid's entire contents (compositor use only)."""
        self.glyph = np.array([[self.registry[c] for c in row] for row in chars], dtype='uint32')
        self.fgcolor = np.asarray(fgcolor, dtype='float32')
        self.bgcolor = np.asarray(bgcolor, dtype='float32')
        self._changed()


class LayerList(object):
    """An ordered, versioned collection of layers (``scene.grids``).

    List order is draw order among screen-space grids (later = on top).
    ``structure_version`` bumps on add/remove so backends can diff the
    membership at sync time; ``changed`` follows the layer contract.
    """
    def __init__(self):
        self._layers = []
        self.structure_version = 0
        self.changed = Observable()

    def __iter__(self):
        return iter(self._layers)

    def __len__(self):
        return len(self._layers)

    def __getitem__(self, i):
        return self._layers[i]

    def add(self, layer):
        self._layers.append(layer)
        self._changed()
        return layer

    def remove(self, layer):
        self._layers.remove(layer)
        self._changed()

    def _changed(self):
        self.structure_version += 1
        self.changed()


class FieldLayer(object):
    """A named float32 scalar/vector field covering the maze (light, LOS, memory, ...)."""
    def __init__(self, name, shape=None, data=None):
        self.name = name
        if data is not None:
            self.data = np.ascontiguousarray(data, dtype='float32')
        else:
            self.data = np.zeros(shape, dtype='float32')
        self.version = 0
        self.changed = Observable()

    def set_data(self, data):
        """Copy *data* into the field (in place when shapes match) and bump version."""
        data = np.asarray(data)
        if data.shape == self.data.shape:
            self.data[...] = data
        else:
            self.data = np.ascontiguousarray(data, dtype='float32')
        self.version += 1
        self.changed()

    def bump(self):
        """Declare that self.data was mutated in place."""
        self.version += 1
        self.changed()
