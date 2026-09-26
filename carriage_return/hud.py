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

Console lines fade with age (turns since written) and salience: see
:func:`brightness`.
"""
import math
import threading
from itertools import islice

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

# Console line fade. A line's brightness starts at b0(salience), interpolated
# linearly from B_LOW to B_HIGH, and decays exponentially with a time constant
# in turns interpolated geometrically from TAU_MIN to TAU_MAX. It is then
# clamped to [BRIGHTNESS_FLOOR, BRIGHTNESS_CEIL]; B_HIGH > BRIGHTNESS_CEIL, so
# high-salience lines hold at full brightness before they start to fade.
B_LOW, B_HIGH = 0.5, 1.5
TAU_MIN, TAU_MAX = 40.0, 400.0
BRIGHTNESS_FLOOR, BRIGHTNESS_CEIL = 0.4, 1.0

# Console scrolling: rows moved per mouse wheel notch, and the scroll bar's colors.
WHEEL_ROWS = 3
SCROLL_TRACK_BG = (0.2, 0.2, 0.25, 1.0)
SCROLL_THUMB_BG = (0.6, 0.6, 0.65, 1.0)


def brightness(age, salience):
    """Console text brightness for a line *age* turns old with *salience* (0-1).

    1.0 is the console's ``FG`` unscaled.
    """
    initial = B_LOW + (B_HIGH - B_LOW) * salience
    tau = TAU_MIN * (TAU_MAX / TAU_MIN) ** salience
    faded = initial * math.exp(-age / tau)
    return min(BRIGHTNESS_CEIL, max(BRIGHTNESS_FLOOR, faded))


def wrap(line, width):
    """Split *line* into chunks of at most *width* characters."""
    if len(line) <= width:
        return [line]
    return [line[i:i + width] for i in range(0, len(line), width)]


class ConsoleWidget(Widget):
    """Renders scene.log into its own cells, newest last, scrollable.

    Each line is drawn at its :func:`brightness` of *fg*. ``scroll`` is how
    many rows the view sits back from the newest line (0 follows the log's
    tail); it returns to 0 whenever an entry is added or removed. With
    *scroll_bar*, the last column holds a scroll bar drawn in cell
    backgrounds.

    Repaints come from whichever thread wrote a message, and scrolling from
    the gameplay thread, which can overlap the command prompt's writes -- so
    both hold a lock.
    """

    def __init__(self, scene, fg=FG, scroll_bar=False):
        self.scene = scene
        self.fg = fg
        self.scroll_bar = scroll_bar
        self.scroll = 0.0
        self._entry_count = len(scene.log.entries)
        self._thumb = None  # (top row, length) of the scroll bar thumb, if any
        self._lock = threading.RLock()
        Widget.__init__(self)
        scene.log.changed.connect(self.repaint)

    def scroll_by(self, rows):
        """Move the view *rows* back into history (negative: toward the newest)."""
        with self._lock:
            self.scroll = max(self.scroll + rows, 0.0)
            self.repaint()

    def scroll_wheel(self, steps):
        """Scroll for *steps* wheel notches (positive: back into history)."""
        self.scroll_by(steps * WHEEL_ROWS)

    def scroll_page(self, pages):
        """Scroll *pages* pages back into history (negative: toward the newest)."""
        self.scroll_by(pages * (self.nrows - 1))

    def click_scroll_bar(self, row, col):
        """Page toward a click on the scroll bar's track, given as a cell of
        the root widget's grid; clicks elsewhere are ignored."""
        row, col = self.local_cell(row, col)
        if (self._thumb is None or col != self.ncols - 1
                or not 0 <= row < self.nrows):
            return
        top, length = self._thumb
        if row < top:
            self.scroll_page(1)
        elif row >= top + length:
            self.scroll_page(-1)

    def _rows_newest_first(self, cols):
        """Yield (text, fg) for each wrapped row of the log, newest first."""
        log = self.scene.log
        for entry in reversed(log.entries):
            k = brightness(log.turn - entry.turn, entry.salience)
            fg = (self.fg[0] * k, self.fg[1] * k, self.fg[2] * k, self.fg[3])
            for chunk in reversed(wrap(entry.text, cols)):
                yield chunk, fg

    def repaint(self):
        """Repaint from the log (runs on whatever thread wrote the message)."""
        with self._lock:
            self._repaint()

    def _repaint(self):
        log = self.scene.log
        if len(log.entries) != self._entry_count:
            self._entry_count = len(log.entries)
            self.scroll = 0.0
        rows = self.nrows
        text_cols = self.ncols - 2 if self.scroll_bar else self.ncols
        back = int(round(self.scroll))
        fetched = list(islice(self._rows_newest_first(text_cols), back + rows))
        if len(fetched) < back + rows:  # scrolled past the oldest row
            back = max(len(fetched) - rows, 0)
            self.scroll = float(back)
        window = fetched[back:]
        self.clear()
        for i, (text, fg) in enumerate(reversed(window), start=rows - len(window)):
            self.write(i, 0, text, fg=fg)
        if self.scroll_bar:
            total = sum(len(wrap(entry.text, text_cols)) for entry in log.entries)
            self._paint_scroll_bar(total, back)

    def _paint_scroll_bar(self, total, back):
        """Draw the scroll bar for a view *back* rows from the newest of *total* rows."""
        rows, col = self.nrows, self.ncols - 1
        self.fill_rect(0, col, rows, 1, bg=SCROLL_TRACK_BG)
        if total <= rows:
            self._thumb = None
            return
        length = max(1, round(rows * rows / total))
        top = round((1 - back / (total - rows)) * (rows - length))
        self._thumb = (top, length)
        self.fill_rect(top, col, length, 1, bg=SCROLL_THUMB_BG)

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

    def console_at(self, event):
        """True if the mouse *event* is over the console panel."""
        return event.grid is self.layer.grid and self.console.contains(event.row, event.col)

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
