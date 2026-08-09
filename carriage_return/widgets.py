"""A small widget tree for the HUD and dialogs (game-side, no rendering library).

``Widget`` is a rectangle of glyph cells positioned relative to a parent,
composited zero-copy: a child's ``glyph``/``fgcolor``/``bgcolor`` arrays are
numpy *views* into its parent's (and transitively the root's) arrays, so a
child's ``write()`` lands directly in the backing array with no explicit
copy-up step. Change notification bubbles the same way: each widget's
``changed`` Observable fires after its own writes, and a parent relays its
children's ``changed`` through its own (see ``add_child``).

``GridLayout`` is a pure geometry helper (Qt ``QGridLayout``/HTML-table
style) used by ``GridFrame``, the border/layout widget that places children
in a grid and draws the outer border plus internal dividers -- with correct
box-drawing junctions where a spanning child swallows a divider, so adjacent
boxes never draw doubled lines.

``WidgetGridLayer`` is the compositor: it bridges one top-level ``Widget``
tree to one real ``CharGridLayer`` in ``scene.grids``, the only thing a
rendering backend ever sees. Grids here are small (tens of rows), so a full
``set_data()`` replace per repaint is cheap -- no per-cell diffing.
"""
import numpy as np

from .events import Observable
from .layers import CharGridLayer


# Shared style defaults; a widget can override per-instance/per-subclass.
FG = (0.9, 0.9, 0.85, 1.0)
BG = (0.05, 0.05, 0.10, 0.92)
BORDER_FG = (1.0, 1.0, 1.0, 0.6)
TITLE_FG = (1.0, 0.95, 0.5, 1.0)
HINT_FG = (0.55, 0.55, 0.55, 1.0)
CURSOR_FG = (0.0, 0.0, 0.0, 1.0)
CURSOR_BG = (0.85, 0.85, 0.65, 0.9)


class Widget(object):
    """A rectangle of glyph cells that knows how to draw itself relative to a parent.

    Owns ``glyph`` (unicode, one char per cell), ``fgcolor``, ``bgcolor``
    (numpy arrays sized to ``(nrows, ncols)``), plus ``row_offset``/
    ``col_offset`` relative to the parent (0, 0 with no parent). A root
    widget (``parent is None``) is constructed with an explicit size; a
    child is sized/positioned by its parent's ``add_child()``.
    """

    def __init__(self, nrows=0, ncols=0):
        self.parent = None
        self.children = []
        self.row_offset = 0
        self.col_offset = 0
        self.version = 0
        self.changed = Observable()
        self._realloc(nrows, ncols)

    @property
    def shape(self):
        return (self.nrows, self.ncols)

    # -- composition ---------------------------------------------------

    def add_child(self, widget, row, col, nrows, ncols):
        """Attach *widget* as a child occupying a sub-rect of this widget.

        Gives the child zero-copy numpy views into this widget's arrays;
        the child's ``changed`` is relayed through this widget's ``changed``
        so a leaf's repaint bubbles to the root one hop per level.
        """
        widget.parent = self
        widget._place(row, col, nrows, ncols)
        self.children.append(widget)
        widget.changed.connect(self._child_changed)
        return widget

    def _place(self, row, col, nrows, ncols):
        """Re-slice this widget's arrays as a view into its parent's current arrays."""
        self.row_offset, self.col_offset = row, col
        self.nrows, self.ncols = nrows, ncols
        parent = self.parent
        self.glyph = parent.glyph[row:row + nrows, col:col + ncols]
        self.fgcolor = parent.fgcolor[row:row + nrows, col:col + ncols]
        self.bgcolor = parent.bgcolor[row:row + nrows, col:col + ncols]
        # this widget's own array identity changed -- refresh grandchildren too
        for child in self.children:
            child._place(child.row_offset, child.col_offset, child.nrows, child.ncols)

    def _move_to(self, row, col, nrows, ncols):
        """Reposition/resize this child within its parent's current arrays (parent-driven relayout)."""
        self._place(row, col, nrows, ncols)
        self._shape_changed()
        self._changed()

    def _child_changed(self):
        self._changed()

    def _changed(self):
        self.version += 1
        self.changed()

    def _realloc(self, nrows, ncols):
        """Allocate fresh backing arrays sized (nrows, ncols) -- root widgets only."""
        self.nrows, self.ncols = nrows, ncols
        self.glyph = np.full((nrows, ncols), ' ', dtype='<U1')
        self.fgcolor = np.tile(np.asarray(FG, dtype='float32'), (nrows, ncols, 1))
        self.bgcolor = np.tile(np.asarray(BG, dtype='float32'), (nrows, ncols, 1))

    # -- drawing ---------------------------------------------------------

    def write(self, row, col, text, fg=None, bg=None):
        """Write *text* on *row* starting at *col*, clipped to the widget.

        *fg*/*bg*, when given, recolor the written cells. One version bump
        per call; writes that are entirely clipped away change nothing and
        do not bump.
        """
        rows, cols = self.nrows, self.ncols
        if not (0 <= row < rows):
            return
        if col < 0:
            text = text[-col:]
            col = 0
        n = min(len(text), cols - col)
        if n <= 0:
            return
        self.glyph[row, col:col + n] = list(text[:n])
        if fg is not None:
            self.fgcolor[row, col:col + n] = fg
        if bg is not None:
            self.bgcolor[row, col:col + n] = bg
        self._changed()

    def fill_row(self, row, fg=None, bg=None):
        """Recolor a full row (cursor highlight bars); glyphs are untouched."""
        if fg is not None:
            self.fgcolor[row, :] = fg
        if bg is not None:
            self.bgcolor[row, :] = bg
        self._changed()

    def clear(self, fg=None, bg=None):
        """Reset every cell to a space, optionally recoloring the whole widget."""
        self.glyph[:] = ' '
        if fg is not None:
            self.fgcolor[:] = fg
        if bg is not None:
            self.bgcolor[:] = bg
        self._changed()

    # -- sizing ------------------------------------------------------------

    def resize(self, nrows, ncols):
        """Resize to (nrows, ncols): called by a parent, or externally for a root.

        Reallocates (root) or re-slices the view from the parent (child, at
        its existing offset), recurses into children, calls the overridable
        ``_shape_changed()`` hook, then repaints.
        """
        if self.parent is None:
            self._realloc(nrows, ncols)
            self._shape_changed()
            self._changed()
        else:
            self._move_to(self.row_offset, self.col_offset, nrows, ncols)

    def _shape_changed(self):
        """Hook for subclasses to react to a size change (e.g. re-wrap text)."""
        pass

    def preferred_shape(self):
        """Return (nrows, ncols) this widget wants given unconstrained space, or None."""
        return None


# -- box-drawing junctions --------------------------------------------------

_JUNCTIONS = {
    (False, False, False, False): ' ',
    (False, False, False, True): '─',
    (False, False, True, False): '─',
    (False, False, True, True): '─',
    (False, True, False, False): '│',
    (True, False, False, False): '│',
    (True, True, False, False): '│',
    (False, True, False, True): '┌',
    (False, True, True, False): '┐',
    (True, False, False, True): '└',
    (True, False, True, False): '┘',
    (False, True, True, True): '┬',
    (True, False, True, True): '┴',
    (True, True, False, True): '├',
    (True, True, True, False): '┤',
    (True, True, True, True): '┼',
}


def junction_char(up, down, left, right):
    """Box-drawing character for an intersection with the given neighboring segments."""
    return _JUNCTIONS[(bool(up), bool(down), bool(left), bool(right))]


def _line_positions(sizes):
    """Pixel row/col index of each border/divider line around a run of *sizes* content cells.

    ``len(sizes) + 1`` positions: index 0 is the outer border before the
    first cell, the last is the outer border after the last cell, and each
    one in between is the one-cell divider after content run *i*.
    """
    positions = [0]
    for size in sizes:
        positions.append(positions[-1] + 1 + size)
    return positions


class GridLayout(object):
    """Pure geometry: row/col sizing, cell rects, and border/divider/junction
    placement for a Qt-``QGridLayout``/HTML-table style grid.

    *row_heights*/*col_widths* are the content sizes (in cells) of each grid
    row/column, not counting the one-cell border/divider lines between and
    around them. *cells* is a list of ``(row, col, rowspan, colspan)`` grid
    placements; a spanning cell swallows the divider(s) inside its span, so
    the border/junction computation below omits them.
    """

    def __init__(self, row_heights, col_widths, cells):
        self.row_heights = list(row_heights)
        self.col_widths = list(col_widths)
        self.cells = list(cells)
        self._row_lines = _line_positions(self.row_heights)
        self._col_lines = _line_positions(self.col_widths)

        n_gr, n_gc = len(self.row_heights), len(self.col_widths)
        occupancy = [[None] * n_gc for _ in range(n_gr)]
        for i, (row, col, rowspan, colspan) in enumerate(self.cells):
            for r in range(row, row + rowspan):
                for c in range(col, col + colspan):
                    occupancy[r][c] = i
        self._occupancy = occupancy

    @property
    def shape(self):
        """Total (nrows, ncols) of the grid, including outer border and dividers."""
        return (self._row_lines[-1] + 1, self._col_lines[-1] + 1)

    @property
    def cell_rects(self):
        """Content rect ``(row_offset, col_offset, nrows, ncols)`` for each cell, in ``cells`` order."""
        rects = []
        for row, col, rowspan, colspan in self.cells:
            row_offset = self._row_lines[row] + 1
            col_offset = self._col_lines[col] + 1
            nrows = self._row_lines[row + rowspan] - self._row_lines[row] - 1
            ncols = self._col_lines[col + colspan] - self._col_lines[col] - 1
            rects.append((row_offset, col_offset, nrows, ncols))
        return rects

    def _h_seg(self, i, c):
        """True if a horizontal segment should be drawn under grid-column *c* along line *i*."""
        n_gr = len(self.row_heights)
        if i == 0 or i == n_gr:
            return True
        above, below = self._occupancy[i - 1][c], self._occupancy[i][c]
        return not (above is not None and above == below)

    def _v_seg(self, r, j):
        """True if a vertical segment should be drawn under grid-row *r* along line *j*."""
        n_gc = len(self.col_widths)
        if j == 0 or j == n_gc:
            return True
        left, right = self._occupancy[r][j - 1], self._occupancy[r][j]
        return not (left is not None and left == right)

    def border_cells(self):
        """List of ``(row, col, char)`` pixel cells for the outer border, internal dividers, and junctions."""
        n_gr, n_gc = len(self.row_heights), len(self.col_widths)
        out = []
        for i in range(n_gr + 1):
            row = self._row_lines[i]
            for c in range(n_gc):
                if self._h_seg(i, c):
                    for col in range(self._col_lines[c] + 1, self._col_lines[c + 1]):
                        out.append((row, col, '─'))
        for j in range(n_gc + 1):
            col = self._col_lines[j]
            for r in range(n_gr):
                if self._v_seg(r, j):
                    for row in range(self._row_lines[r] + 1, self._row_lines[r + 1]):
                        out.append((row, col, '│'))
        for i in range(n_gr + 1):
            for j in range(n_gc + 1):
                up = self._v_seg(i - 1, j) if i > 0 else False
                down = self._v_seg(i, j) if i < n_gr else False
                left = self._h_seg(i, j - 1) if j > 0 else False
                right = self._h_seg(i, j) if j < n_gc else False
                if up or down or left or right:
                    out.append((self._row_lines[i], self._col_lines[j],
                                junction_char(up, down, left, right)))
        return out


class GridFrame(Widget):
    """A ``Widget`` that lays out children in a grid and draws the border/dividers.

    Serves both the HUD container (stats/info/console in one frame with no
    doubled seams) and a dialog's outer border (a 1x1 grid around one
    content widget).
    """

    def __init__(self, row_heights, col_widths, cells, fg=None, bg=None, border_fg=None):
        """*cells*: list of ``(widget, row, col, rowspan, colspan)`` grid placements."""
        self._row_heights = list(row_heights)
        self._col_widths = list(col_widths)
        self._cells = list(cells)
        self.fg = FG if fg is None else fg
        self.bg = BG if bg is None else bg
        self.border_fg = BORDER_FG if border_fg is None else border_fg
        layout = self._build_layout()
        Widget.__init__(self, *layout.shape)
        self._shape_changed()

    def _build_layout(self):
        return GridLayout(self._row_heights, self._col_widths,
                          [(row, col, rowspan, colspan)
                           for (_, row, col, rowspan, colspan) in self._cells])

    def set_layout(self, row_heights, col_widths):
        """Replace the row/column sizing and rebuild the grid (children keep their assigned cells)."""
        self._row_heights = list(row_heights)
        self._col_widths = list(col_widths)
        self.resize(*self._build_layout().shape)

    def _shape_changed(self):
        layout = self._build_layout()
        for (widget, _row, _col, _rowspan, _colspan), rect in zip(self._cells, layout.cell_rects):
            if widget.parent is self:
                widget._move_to(*rect)
            else:
                self.add_child(widget, *rect)
        self.clear(fg=self.fg, bg=self.bg)
        for row, col, char in layout.border_cells():
            self.write(row, col, char, fg=self.border_fg)


class WidgetGridLayer(object):
    """Bridges one top-level ``Widget`` tree to one ``CharGridLayer`` in ``scene.grids``.

    One ``WidgetGridLayer`` per independent on-screen layer: the HUD is one,
    and each open dialog gets its own -- matching ``scene.grids``' existing
    "later = drawn on top" ordering for popups over the HUD.
    """

    def __init__(self, scene, root_widget, anchor='center', offset=(0, 0), name=None):
        self.scene = scene
        self.root_widget = root_widget
        self.grid = CharGridLayer(scene.glyphs, root_widget.shape, space='screen',
                                  anchor=anchor, offset=offset, name=name)
        scene.grids.add(self.grid)
        root_widget.changed.connect(self._repaint)
        self._repaint()

    def _repaint(self):
        self.grid.set_data(self.root_widget.glyph, self.root_widget.fgcolor, self.root_widget.bgcolor)

    def resize(self, shape):
        """Resize the root widget then the backing ``CharGridLayer`` to match."""
        self.root_widget.resize(*shape)
        self.grid.reshape(shape)
        self._repaint()

    def sync(self):
        """Match the ``CharGridLayer`` to ``root_widget``'s current shape and repaint.

        For when the root widget's own layout already changed shape on its
        own terms (e.g. a ``GridFrame.set_layout()`` call, which recomputes
        row/column sizing that ``resize(shape)`` alone can't express) --
        reshapes the backing grid to match rather than driving the resize
        from this end.
        """
        self.grid.reshape(self.root_widget.shape)
        self._repaint()

    def close(self):
        """Remove the grid from the scene and stop observing the widget tree."""
        self.root_widget.changed.disconnect(self._repaint)
        self.scene.grids.remove(self.grid)
