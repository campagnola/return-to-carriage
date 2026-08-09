"""The Pager dialog ("book"): interactive state, key loop, and drawing in one widget.

``PagerWidget`` is a ``carriage_return.widgets.Widget`` holding multi-page
text (a book, a scroll) viewed one page at a time, plus its own rendering.
``run_pager`` is the sequential key loop a DialogSession runs on its own
thread. See ``dialogs/__init__.py`` for how a PagerWidget is wrapped in a
bordered ``GridFrame`` and composited to the screen.
"""
from ..input import KeyPress
from ..widgets import FG, BG, HINT_FG, TITLE_FG, Widget


class PagerWidget(Widget):
    """Multi-page text (a book, a scroll) viewed one page at a time.

    ``pages`` is a list of strings, one per page (each possibly multi-line).
    Every mutator repaints the widget's own cells before returning, so a
    caller never needs to trigger a separate render pass.
    """

    MIN_COLS = 30

    def __init__(self, title, pages):
        Widget.__init__(self)
        self.title = title
        self.pages = list(pages)
        self.page = 0
        self.done = False
        self.result = None

    @property
    def page_count(self):
        return len(self.pages)

    @property
    def page_text(self):
        """Text of the current page ('' for an empty pager)."""
        if len(self.pages) == 0:
            return ''
        return self.pages[self.page]

    def _set_page(self, page):
        page = min(max(page, 0), max(len(self.pages) - 1, 0))
        if page != self.page:
            self.page = page
            self._changed()
            self.repaint()

    def next_page(self):
        self._set_page(self.page + 1)

    def prev_page(self):
        self._set_page(self.page - 1)

    def close(self):
        if self.done:
            return
        self.done = True
        self._changed()
        self.repaint()

    # -- sizing/drawing ------------------------------------------------

    def _footer(self):
        return "page %d/%d   Left/Right flip  Esc close" % (
            self.page + 1, max(self.page_count, 1))

    def preferred_shape(self):
        """(nrows, ncols) this pager wants: title/blank/page body/blank/footer, capped at 100 cols."""
        page_lines = [page.splitlines() or [''] for page in self.pages] or [['']]
        rows = max(len(lines) for lines in page_lines)
        widths = [len(line) for lines in page_lines for line in lines]
        widths += [len(self.title), len(self._footer()), self.MIN_COLS]
        return rows + 4, min(max(widths), 100)

    def _shape_changed(self):
        self.repaint()

    def repaint(self):
        """Redraw the whole widget from current state (title, current page, footer)."""
        self.clear(fg=FG, bg=BG)
        self.write(0, 0, self.title, fg=TITLE_FG)
        for i, line in enumerate(self.page_text.splitlines()):
            if 2 + i >= self.nrows - 2:
                break
            self.write(2 + i, 0, line)
        self.write(self.nrows - 1, 0, self._footer(), fg=HINT_FG)


def run_pager(session, pager):
    """Standard key handling for a Pager; returns None when closed.

    Right/Down flip forward, Left/Up flip back, Escape/Enter close.
    """
    while True:
        event = session.get()
        if not isinstance(event, KeyPress):
            continue
        if event.key in ('Right', 'Down'):
            pager.next_page()
        elif event.key in ('Left', 'Up'):
            pager.prev_page()
        elif event.key in ('Escape', 'Enter', 'Return'):
            pager.close()
            return None
