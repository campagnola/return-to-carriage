"""Game-side HUD: stats bar, info box, and message console as one widget tree.

The HUD is a single ``carriage_return.widgets.GridFrame`` (stats spanning the
top, info and console splitting the row below) wrapped in one
``WidgetGridLayer``, which bridges it to one real ``CharGridLayer`` in
``scene.grids``. Drawing the whole frame -- outer border, the stats/info
divider, and the info/console divider -- from a single owning widget means
adjacent boxes never draw doubled border lines, unlike the old per-box
painters this replaces.

Layout approximates the previous fixed vispy widget grid: a full-width stats
bar sitting on top of an info box (bottom-left) and the message console
(bottom-right). Column widths derive from ``scene.screen`` (the canvas size
in cells, backend-written): the Hud observes it and relays out the frame's
columns -- with text re-wrapped -- whenever the window is resized.

The console repaints whenever ``scene.log`` changes -- ConsoleWidget
subscribes to its ``changed`` event.
"""
from .widgets import GridFrame, Widget, WidgetGridLayer

# placeholder content, hard-coded until real stats/inspection exist
STATS_TEXT = ("HP:17/33   Food:56%  Water:34%  Sleep:65%   Weight:207(45)"
              "    Level:3  Int:12  Str:9  Wis:11  Cha:2")
INFO_TEXT = ""

# HUD-specific style: dimmer/more translucent than the dialog boxes
# (widgets.FG/BG/BORDER_FG), matching the old ConsolePainter/StaticTextPainter
# look.
FG = (1.0, 1.0, 1.0, 0.5)
BG = (0.0, 0.0, 0.0, 0.4)
BORDER_FG = (1.0, 1.0, 1.0, 0.3)


def wrap(line, width):
    """Split *line* into chunks of at most *width* characters."""
    if len(line) <= width:
        return [line]
    return [line[i:i + width] for i in range(0, len(line), width)]


class ConsoleWidget(Widget):
    """Renders the tail of scene.log into its own cells, newest last."""

    def __init__(self, scene):
        self.scene = scene
        Widget.__init__(self)
        scene.log.changed.connect(self.repaint)

    def repaint(self):
        """Repaint from the log tail (runs on whatever thread wrote the message)."""
        rows, cols = self.nrows, self.ncols
        self.clear()
        lines = []
        for line in self.scene.log.lines[-rows:]:
            lines.extend(wrap(line, cols))
        tail = lines[-rows:]
        for i, line in enumerate(tail, start=rows - len(tail)):
            self.write(i, 0, line)

    def _shape_changed(self):
        self.repaint()

    def close(self):
        """Stop observing the log."""
        self.scene.log.changed.disconnect(self.repaint)


class TextWidget(Widget):
    """A fixed block of text wrapped to the widget's width (stats bar, info box)."""

    def __init__(self, text=''):
        self._text = text
        Widget.__init__(self)

    def set_text(self, text):
        """Replace the displayed text and repaint."""
        self._text = text
        self._paint()

    def _shape_changed(self):
        self._paint()

    def _paint(self):
        self.clear()
        lines = []
        for line in self._text.split('\n'):
            lines.extend(wrap(line, self.ncols))
        for i, line in enumerate(lines[:self.nrows]):
            self.write(i, 0, line)


class Hud(object):
    """The stats/info/console HUD, laid out in one bordered GridFrame.

    The info box (bottom-left) and console (bottom-right) split the bottom
    width of the canvas; the stats bar spans the full width directly above
    them, sharing borders with both so there are no doubled seams. The Hud
    subscribes to ``scene.screen.changed``: a resize relays out the frame's
    columns (the info/console split follows the screen width) and re-wraps
    text.
    """
    box_rows = 12       # info box and console height, border ring included
    stats_rows = 3      # one text row + border ring
    info_frac = 0.4     # share of the bottom width given to the info box

    def __init__(self, scene):
        self.scene = scene
        self.stats = TextWidget()
        self.info = TextWidget()
        self.console = ConsoleWidget(scene)
        row_heights, col_widths = self._sizes()
        self.frame = GridFrame(row_heights, col_widths, [
            (self.stats, 0, 0, 1, 2),
            (self.info, 1, 0, 1, 1),
            (self.console, 1, 1, 1, 1),
        ], fg=FG, bg=BG, border_fg=BORDER_FG)
        # GridFrame's initial layout pass places children via add_child(),
        # which doesn't invoke their _shape_changed() hook -- paint their
        # starting content explicitly now that they have real cells.
        self.stats.set_text(STATS_TEXT)
        self.info.set_text(INFO_TEXT)
        self.console.repaint()
        self.layer = WidgetGridLayer(scene, self.frame, anchor='bottom-left')
        scene.screen.changed.connect(self._screen_changed)

    def _sizes(self):
        """(row_heights, col_widths) content sizes for the current screen width."""
        _, cols = self.scene.screen.shape
        content_width = max(cols - 3, 2)  # minus outer border cols + divider
        info_cols = max(int(round(content_width * self.info_frac)), 1)
        console_cols = max(content_width - info_cols, 1)
        return ([self.stats_rows - 2, self.box_rows - 2], [info_cols, console_cols])

    def _screen_changed(self):
        row_heights, col_widths = self._sizes()
        self.frame.set_layout(row_heights, col_widths)
        self.layer.sync()

    def close(self):
        self.scene.screen.changed.disconnect(self._screen_changed)
        self.console.close()
        self.layer.close()


def build_hud(scene):
    """Create the standard HUD for *scene*; returns the Hud."""
    return Hud(scene)
