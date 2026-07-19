"""Game-side HUD painters: console, stats, and info as CharGridLayers.

Each painter owns one screen-space grid in ``scene.grids`` and fills it from
game state; rendering backends draw the grids generically (they cannot tell
a console from a menu). The console repaints whenever ``scene.log`` changes
— the ConsolePainter subscribes to its ``changed`` event.

Layout approximates the old fixed vispy widget grid: a full-width stats bar
sitting on top of an info box (bottom-left) and the message console
(bottom-right). Shapes derive from ``scene.screen`` (the canvas size in
cells, backend-written): the Hud observes it and reshapes its grids — with
text re-wrapped — whenever the window is resized.

Each box paints a one-cell character border ring (shared with the dialog
painters); text fills the cells inside it.
"""
from .dialogs.base import draw_border
from .layers import CharGridLayer

# placeholder content, hard-coded until real stats/inspection exist
STATS_TEXT = ("HP:17/33   Food:56%  Water:34%  Sleep:65%   Weight:207(45)"
              "    Level:3  Int:12  Str:9  Wis:11  Cha:2")
INFO_TEXT = "There is a scroll of infinite recursion here."


def wrap(line, width):
    """Split *line* into chunks of at most *width* characters."""
    if len(line) <= width:
        return [line]
    return [line[i:i + width] for i in range(0, len(line), width)]


class ConsolePainter(object):
    """Renders the tail of scene.log into a bottom-right grid, newest last."""

    FG = (1.0, 1.0, 1.0, 0.5)
    BG = (0.0, 0.0, 0.0, 0.4)
    BORDER_FG = (1.0, 1.0, 1.0, 0.3)

    def __init__(self, scene, shape):
        self.scene = scene
        self.grid = CharGridLayer(scene.glyphs, shape, space='screen',
                                  anchor='bottom-right', name='console')
        scene.log.changed.connect(self.paint)
        scene.grids.add(self.grid)
        self.paint()

    def set_shape(self, shape):
        """Reshape the grid (window resize) and repaint, re-wrapping the log."""
        self.grid.reshape(shape)
        self.paint()

    def paint(self):
        """Repaint from the log (runs on whatever thread wrote the message)."""
        rows, cols = self.grid.shape
        content_rows, content_cols = rows - 2, cols - 2
        self.grid.clear(fg=self.FG, bg=self.BG)
        draw_border(self.grid, self.BORDER_FG)
        lines = []
        for line in self.scene.log.lines[-content_rows:]:
            lines.extend(wrap(line, content_cols))
        tail = lines[-content_rows:]
        for i, line in enumerate(tail, start=rows - 1 - len(tail)):
            self.grid.write(i, 1, line)

    def close(self):
        self.scene.log.changed.disconnect(self.paint)
        self.scene.grids.remove(self.grid)


class StaticTextPainter(object):
    """A fixed block of text in a bordered grid (stats bar, info box)."""

    FG = (1.0, 1.0, 1.0, 0.5)
    BG = (0.0, 0.0, 0.0, 0.4)
    BORDER_FG = (1.0, 1.0, 1.0, 0.3)

    def __init__(self, scene, shape, text, anchor, offset=(0, 0), name=None):
        self.scene = scene
        self.grid = CharGridLayer(scene.glyphs, shape, space='screen',
                                  anchor=anchor, offset=offset, name=name)
        scene.grids.add(self.grid)
        self.set_text(text)

    def set_text(self, text):
        self._text = text
        self._paint()

    def set_shape(self, shape):
        """Reshape the grid (window resize) and repaint, re-wrapping the text."""
        self.grid.reshape(shape)
        self._paint()

    def _paint(self):
        rows, cols = self.grid.shape
        self.grid.clear(fg=self.FG, bg=self.BG)
        draw_border(self.grid, self.BORDER_FG)
        lines = []
        for line in self._text.split('\n'):
            lines.extend(wrap(line, cols - 2))
        for i, line in enumerate(lines[:rows - 2]):
            self.grid.write(i + 1, 1, line)

    def close(self):
        self.scene.grids.remove(self.grid)


class Hud(object):
    """The three HUD painters, laid out against ``scene.screen``.

    The info box (bottom-left) and console (bottom-right) split the bottom
    rows of the canvas; the stats bar spans the full width directly above
    them. The Hud subscribes to ``scene.screen.changed``: a resize reshapes
    all three grids and re-wraps their text.
    """
    box_rows = 12       # info box and console height
    stats_rows = 3      # one text row + border ring
    info_frac = 0.4     # share of the bottom width given to the info box

    def __init__(self, scene):
        self.scene = scene
        info_shape, console_shape, stats_shape = self._shapes()
        self.info = StaticTextPainter(scene, info_shape, INFO_TEXT,
                                      anchor='bottom-left', name='info')
        self.console = ConsolePainter(scene, console_shape)
        self.stats = StaticTextPainter(scene, stats_shape, STATS_TEXT,
                                       anchor='bottom-left',
                                       offset=(self.box_rows, 0), name='stats')
        scene.screen.changed.connect(self._screen_changed)

    def _shapes(self):
        """Layout: (info, console, stats) shapes for the current screen."""
        rows, cols = self.scene.screen.shape
        info_cols = max(int(round(cols * self.info_frac)), 3)
        console_cols = max(cols - info_cols, 3)
        return ((self.box_rows, info_cols),
                (self.box_rows, console_cols),
                (self.stats_rows, cols))

    def _screen_changed(self):
        info_shape, console_shape, stats_shape = self._shapes()
        self.info.set_shape(info_shape)
        self.console.set_shape(console_shape)
        self.stats.set_shape(stats_shape)

    def close(self):
        self.scene.screen.changed.disconnect(self._screen_changed)
        self.info.close()
        self.console.close()
        self.stats.close()


def build_hud(scene):
    """Create the standard HUD painters for *scene*; returns the Hud."""
    return Hud(scene)
