"""The message-log dialog: a large scrollable view of ``scene.log``.

``LogDialog`` is a header row over a scroll-barred ``ConsoleWidget``, framed
and centered in a ``WidgetGridLayer`` that covers most of the screen and
follows window resizes. ``run_log`` is the sequential input loop a
DialogSession runs on its own thread. The log's wrapping, fading and
scrolling are the ``ConsoleWidget``'s.
"""
from ..hud import ConsoleWidget
from ..input import KeyPress, MouseClick, MouseWheel
from ..widgets import FG, HINT_FG, TITLE_FG, GridFrame, Widget, WidgetGridLayer

MARGIN_ROWS, MARGIN_COLS = 2, 3  # free screen cells around the dialog frame
TITLE = "Message log"
HINT = "Wheel/Up/Down/PgUp/PgDn scroll  Esc close"


class HeaderWidget(Widget):
    """The dialog's title and key hints, on one row."""

    def _shape_changed(self):
        self.repaint()

    def repaint(self):
        with self.batched():
            self.clear()
            self.write(0, 0, TITLE, fg=TITLE_FG)
            self.write(0, len(TITLE) + 3, HINT, fg=HINT_FG)


class LogDialog(object):
    """The log dialog's widget tree and its layer in ``scene.grids``."""

    def __init__(self, scene):
        self.scene = scene
        self.header = HeaderWidget()
        self.view = ConsoleWidget(scene, fg=FG, scroll_bar=True)
        row_heights, col_widths = self._sizes()
        self.frame = GridFrame(row_heights, col_widths, [
            (self.header, 0, 0, 1, 1),
            (self.view, 1, 0, 1, 1),
        ])
        # GridFrame's initial layout pass places children via add_child(),
        # which doesn't invoke their _shape_changed() hook (same as Hud).
        self.header.repaint()
        self.view.repaint()
        self.layer = WidgetGridLayer(scene, self.frame, anchor='center')
        scene.screen.changed.connect(self._screen_changed)

    def _sizes(self):
        """(row_heights, col_widths) content sizes for the current screen size."""
        rows, cols = self.scene.screen.shape
        # frame = content + outer border (2) [+ header/log divider (1)]
        log_rows = max(rows - 2 * MARGIN_ROWS - 4, 1)
        log_cols = max(cols - 2 * MARGIN_COLS - 2, len(TITLE))
        return [1, log_rows], [log_cols]

    def _screen_changed(self):
        self.frame.set_layout(*self._sizes())
        self.layer.sync()

    def close(self):
        self.scene.screen.changed.disconnect(self._screen_changed)
        self.view.close()
        self.layer.close()


def run_log(session, dialog):
    """Standard input handling for the log dialog; returns None when closed.

    The wheel and Up/Down/PageUp/PageDown scroll, a click on the scroll bar
    pages toward it, Escape/Enter close.
    """
    view = dialog.view
    while True:
        event = session.get()
        if isinstance(event, MouseWheel):
            view.scroll_wheel(event.steps)
        elif isinstance(event, MouseClick):
            if event.grid is dialog.layer.grid:
                view.click_scroll_bar(event.row, event.col)
        elif isinstance(event, KeyPress):
            if event.key == 'Up':
                view.scroll_by(1)
            elif event.key == 'Down':
                view.scroll_by(-1)
            elif event.key == 'PageUp':
                view.scroll_page(1)
            elif event.key == 'PageDown':
                view.scroll_page(-1)
            elif event.key in ('Escape', 'Enter', 'Return'):
                return None
