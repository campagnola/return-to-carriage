"""The Pager dialog ("book"): model, key loop, and painter in one place.

``Pager`` is a pure-data model of multi-page text; ``run_pager`` is the
sequential key loop a DialogSession runs on its own thread; ``PagerPainter``
renders the model into a screen-space CharGridLayer. See base.py for the
change-tracking and painter contracts.
"""
from ..input import KeyPress
from .base import CharGridPainter, Widget


class Pager(Widget):
    """Multi-page text (a book, a scroll) viewed one page at a time.

    ``pages`` is a list of strings, one per page (each possibly multi-line).
    """

    def __init__(self, title, pages):
        Widget.__init__(self)
        self.title = title
        self.pages = list(pages)
        self.page = 0

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

    def next_page(self):
        self._set_page(self.page + 1)

    def prev_page(self):
        self._set_page(self.page - 1)

    def close(self):
        if self.done:
            return
        self.done = True
        self._changed()


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


class PagerPainter(CharGridPainter):
    """Paints a Pager: title, one page of text, page-count footer."""

    MIN_COLS = 30

    def _footer(self):
        pager = self.model
        return "page %d/%d   Left/Right flip  Esc close" % (
            pager.page + 1, max(pager.page_count, 1))

    def _content_shape(self):
        pager = self.model
        page_lines = [page.splitlines() or [''] for page in pager.pages] or [['']]
        rows = max(len(lines) for lines in page_lines)
        widths = [len(line) for lines in page_lines for line in lines]
        widths += [len(pager.title), len(self._footer()), self.MIN_COLS]
        # title, blank, page body, blank, footer
        return rows + 4, min(max(widths), 100)

    def _render_content(self):
        pager = self.model
        self._write_line(0, pager.title, fg=self.TITLE_FG)
        for i, line in enumerate(pager.page_text.splitlines()):
            if 2 + i >= self.nrows - 2:
                break
            self._write_line(2 + i, line)
        self._write_line(self.nrows - 1, self._footer(), fg=self.HINT_FG)
