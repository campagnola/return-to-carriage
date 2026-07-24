"""Per-slot color modifiers: additive deltas composed onto base sprite colors.

ColorModifier holds deltas for any of the four color channels (fg_emission,
bg_emission, fgcolor, bgcolor).  Absent channels (None) contribute nothing.
SpriteSlot.add_modifier / remove_modifier attach and detach modifiers;
SpriteSlot._recompose writes base + Σ modifiers into the layer arrays.
"""

import numpy as np


class ColorModifier:
    """Additive glyph color delta attached to one or more cells in a SpriteSlot.

    Holds deltas for any of the four channels (fg_emission, bg_emission,
    fgcolor, bgcolor). Absent channels contribute nothing. Mutable — update
    the delta fields directly; the owning slot will apply them on the next
    recompose pass.

    Usage:
        mod = ColorModifier()          # targets all cells in the slot
        mod.fg_emission = np.array([1.0, 0.5, 0.0])
        slot.add_modifier(mod)
    """
    def __init__(self, cells=None):
        # cells: index or slice into the slot's cell range (None → all cells)
        self.cells = slice(None) if cells is None else cells
        self.fg_emission = None   # np.ndarray shape (3,) or (N, 3) or None
        self.bg_emission = None
        self.fgcolor = None       # np.ndarray shape (4,) or (N, 4) or None
        self.bgcolor = None
        self._slot = None         # set by SpriteSlot.add_modifier
