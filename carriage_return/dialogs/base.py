"""Shared dialog foundations: the Widget model base and the painter base.

``Widget`` holds the change-tracking contract every dialog model shares (a
``version`` counter bumped on real state changes plus a single-slot
``observer`` callback). ``CharGridPainter`` owns a screen-space CharGridLayer
in ``scene.grids`` and renders a bordered, padded box around content a
subclass draws — everything visual (border, title, cursor bar, hints) is
written into the grid as characters and cell colors, so rendering backends
draw the grid without knowing what it means.

Both are game-side and numpy-only. A painter installs itself as its model's
``observer`` and repaints the whole grid on any change; that runs on whatever
thread mutates the model (dialog threads, per the session.py contract) and is
pure numpy. Repainting everything per change is deliberate — these grids are
tens of rows at most.
"""
from ..layers import CharGridLayer


def draw_border(grid, fg, bg=None):
    """Write a one-cell '+--+' character border ring on *grid*'s outer cells.

    Shared by the dialog painters here and the HUD painters (hud.py) —
    borders are just cell data, so any bordered box draws them this way.
    """
    rows, cols = grid.shape
    bar = '+' + '-' * (cols - 2) + '+'
    grid.write(0, 0, bar, fg=fg, bg=bg)
    grid.write(rows - 1, 0, bar, fg=fg, bg=bg)
    for row in range(1, rows - 1):
        grid.write(row, 0, '|', fg=fg, bg=bg)
        grid.write(row, cols - 1, '|', fg=fg, bg=bg)


class Widget(object):
    """Base for dialog widget models: version counter + observer + done flag."""

    def __init__(self):
        self.version = 0
        self.observer = None
        self.done = False
        self.result = None

    def _changed(self):
        self.version += 1
        if self.observer is not None:
            self.observer()


class CharGridPainter(object):
    """Base for dialog painters: a bordered, padded grid around content.

    Subclasses declare content size via ``_content_shape()`` -> (rows, cols)
    and fill the content area in ``_render_content()`` using ``_write_line``
    / ``_fill_row``. The full grid is content + ``padding`` blank cells + a
    one-cell character border.
    """
    padding = 1              # blank cells between border and content

    FG = (0.9, 0.9, 0.85, 1.0)
    BG = (0.05, 0.05, 0.10, 0.92)
    BORDER_FG = (1.0, 1.0, 1.0, 0.6)
    TITLE_FG = (1.0, 0.95, 0.5, 1.0)
    HINT_FG = (0.55, 0.55, 0.55, 1.0)
    CURSOR_FG = (0.0, 0.0, 0.0, 1.0)
    CURSOR_BG = (0.85, 0.85, 0.65, 0.9)

    def __init__(self, scene, model, anchor='center', offset=(0, 0)):
        self.scene = scene
        self.model = model
        self.nrows, self.ncols = self._content_shape()
        margin = self.padding + 1  # padding + border ring
        grid_shape = (self.nrows + 2 * margin, self.ncols + 2 * margin)
        self.grid = CharGridLayer(scene.glyphs, grid_shape, space='screen',
                                  anchor=anchor, offset=offset)
        self._observer = self.paint
        model.observer = self._observer
        scene.grids.add(self.grid)
        self.paint()

    # -- content -----------------------------------------------------------

    def _content_shape(self):
        raise NotImplementedError()

    def _render_content(self):
        raise NotImplementedError()

    def paint(self):
        """Repaint the whole grid from the model (any mutating thread)."""
        self.grid.clear(fg=self.FG, bg=self.BG)
        self._draw_border()
        self._render_content()

    def _draw_border(self):
        draw_border(self.grid, self.BORDER_FG)

    def _write_line(self, row, text, fg=None):
        """Write *text* on content row *row* (0 = top), clipped to the content."""
        margin = self.padding + 1
        self.grid.write(row + margin, margin, text[:self.ncols], fg=fg)

    def _fill_row(self, row, fg, bg):
        """Recolor a full content row (cursor highlight bar), keeping the border."""
        margin = self.padding + 1
        grid_row = row + margin
        self.grid.fill_row(grid_row, fg=fg, bg=bg)
        # restore the border cells the fill just recolored
        self.grid.write(grid_row, 0, '|', fg=self.BORDER_FG, bg=self.BG)
        self.grid.write(grid_row, self.grid.shape[1] - 1, '|',
                        fg=self.BORDER_FG, bg=self.BG)

    # -- lifecycle ----------------------------------------------------------

    def close(self):
        """Remove the grid from the scene and stop observing the model."""
        if self.model.observer is self._observer:
            self.model.observer = None
        self.scene.grids.remove(self.grid)
