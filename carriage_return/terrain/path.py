"""Meandering paths: the shape a level's dirt paths and rivers alike take
between two points, kept generic here so a level only has to decide *where*
one runs (see a level's own placement code, e.g. ``levels/level_001_home.py``)
and terrain decides its exact wandering shape and how it paints onto a maze.
"""
import numpy as np

from .meander import meander


class Path:
    """A meandering line of cells across the terrain.

    Indexed along *axis* (``'x'`` or ``'y'``) from *lo* through
    ``lo + len(centerline) - 1``; ``centerline[i]`` is the cell's coordinate
    on the *other* axis at index ``lo + i``. *width* cells are painted
    across the centreline (see :meth:`mask`/:meth:`paint`) with
    *blocktype_id*.

    Kept around after painting -- rather than discarded -- so that later
    code can work from its geometry directly instead of re-deriving the
    curve or scanning the maze: finding where two paths cross, or placing a
    building a few cells off to one side (see :meth:`point_at`).
    """
    def __init__(self, axis, lo, centerline, width, blocktype_id):
        assert axis in ('x', 'y')
        self.axis = axis
        self.lo = lo
        self.centerline = np.asarray(centerline)
        self.width = width
        # the band painted at a given coord runs [center - half, center - half
        # + width) -- centred on the centreline (up to rounding for an even
        # width). A single, named source of truth for that offset, since
        # anything that needs to know exactly which cells got painted (e.g.
        # finding where two paths cross) has to agree with _stamp about it.
        self.half = width // 2
        self.blocktype_id = blocktype_id

    @property
    def hi(self):
        return self.lo + len(self.centerline) - 1

    def center_at(self, coord):
        """The perpendicular-axis coordinate at *coord* along this path's own axis."""
        return int(self.centerline[coord - self.lo])

    def point_at(self, coord, offset=0):
        """The ``(x, y)`` point on the path, *offset* cells to one side, at *coord*."""
        c = self.center_at(coord) + offset
        return (coord, c) if self.axis == 'x' else (c, coord)

    def band_at(self, coord):
        """The half-open ``(start, stop)`` slice on the perpendicular axis
        that this path's *width*-wide band actually paints at *coord* (see
        :meth:`paint`)."""
        c = self.center_at(coord)
        return c - self.half, c - self.half + self.width

    def trimmed(self, hi):
        """A copy of this path, shortened to end at *hi* (inclusive)."""
        return Path(self.axis, self.lo, self.centerline[:hi - self.lo + 1], self.width,
                    self.blocktype_id)

    def mask(self, shape):
        """A boolean *shape* ``(rows, cols)`` array of every cell this path covers."""
        m = np.zeros(shape, dtype=bool)
        self._stamp(m, True)
        return m

    def paint(self, maze):
        """Stamp *blocktype_id* onto *maze.blocks* along this path."""
        self._stamp(maze.blocks, self.blocktype_id)

    def _stamp(self, array, value):
        for i, c in enumerate(self.centerline):
            coord = self.lo + i
            band_lo = c - self.half
            if self.axis == 'x':
                array[band_lo:band_lo + self.width, coord] = value
            else:
                array[coord, band_lo:band_lo + self.width] = value


def create_path(rng, start, end, amplitude, wavelength, width, blocktype_id, bounds=None):
    """Build a meandering :class:`Path` of *blocktype_id* from *start* to
    *end* (each an ``(x, y)`` pair), and return it -- the caller decides
    when (or whether) to :meth:`~Path.paint` it.

    The path runs along whichever axis has the larger span between the
    endpoints, wandering in the other axis via :func:`~.meander.meander`
    (drifting from start's coordinate on that axis to end's, so start and
    end need not share it -- a fixed centre, the common case, is just the
    special case where they're equal). *bounds*, if given, is the ``(lo,
    hi)`` range on the wandering axis that the *painted band* must stay
    within (the centreline itself is clipped a half-width in from each, to
    account for *width*).
    """
    x0, y0 = start
    x1, y1 = end
    if abs(x1 - x0) >= abs(y1 - y0):
        axis = 'x'
        lo, hi = min(x0, x1), max(x0, x1)
        c0, c1 = (y0, y1) if x0 <= x1 else (y1, y0)
    else:
        axis = 'y'
        lo, hi = min(y0, y1), max(y0, y1)
        c0, c1 = (x0, x1) if y0 <= y1 else (x1, y0)

    centerline = meander(rng, hi - lo + 1, c0, c1, amplitude, wavelength)
    if bounds is not None:
        half = width // 2
        lo_b, hi_b = bounds[0] + half, bounds[1] - (width - half - 1)
        centerline = np.clip(centerline, lo_b, hi_b)
    return Path(axis, lo, centerline, width, blocktype_id)
