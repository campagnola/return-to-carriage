"""The Menu dialog: model, key loop, and painter in one place.

``Menu``/``MenuItem`` are pure-data models (a list of selectable items with a
cursor, optionally with checkboxes); ``run_menu`` is the sequential key loop a
DialogSession runs on its own thread; ``MenuPainter`` renders the model into a
screen-space CharGridLayer. See base.py for the change-tracking and painter
contracts.
"""
from ..input import KeyPress
from .base import CharGridPainter, Widget


class MenuItem(object):
    """One entry in a Menu: a display label, an arbitrary value, a checked flag.

    ``value`` is what accept() returns for this item; it defaults to the label
    when not given.
    """

    def __init__(self, label, value=None, checked=False):
        self.label = label
        self.value = label if value is None else value
        self.checked = checked

    def __repr__(self):
        return "<MenuItem %r checked=%r>" % (self.label, self.checked)


class Menu(Widget):
    """A list of selectable items with a cursor, optionally with checkboxes.

    Items may be given as MenuItem instances or plain strings (auto-wrapped).
    In multi_select mode toggle() checks/unchecks the item under the cursor
    and accept() returns the list of checked items' values (falling back to
    the item under the cursor when nothing is checked). In single-select mode
    accept() returns the current item's value.

    accept() and cancel() set ``done`` (and ``result``) so a renderer can
    observe closure; they are idempotent once done.
    """

    def __init__(self, title, items, multi_select=False):
        Widget.__init__(self)
        self.title = title
        self.items = [item if isinstance(item, MenuItem) else MenuItem(item)
                      for item in items]
        self.multi_select = multi_select
        self.cursor = 0

    def __len__(self):
        return len(self.items)

    @property
    def current(self):
        """The MenuItem under the cursor, or None for an empty menu."""
        if len(self.items) == 0:
            return None
        return self.items[self.cursor]

    def move(self, delta):
        """Move the cursor by *delta*, clamped to the item range."""
        if len(self.items) == 0:
            return
        cursor = min(max(self.cursor + delta, 0), len(self.items) - 1)
        if cursor != self.cursor:
            self.cursor = cursor
            self._changed()

    def toggle(self):
        """Toggle the checkbox under the cursor (no-op unless multi_select)."""
        if not self.multi_select or len(self.items) == 0:
            return
        item = self.items[self.cursor]
        item.checked = not item.checked
        self._changed()

    def checked_items(self):
        return [item for item in self.items if item.checked]

    def accept(self):
        """Close the menu, returning (and storing in ``result``) the selection.

        multi_select: list of checked items' values; if none are checked, the
        item under the cursor ([] for an empty menu). Single-select: the
        current item's value (None for an empty menu).
        """
        if self.done:
            return self.result
        if self.multi_select:
            checked = self.checked_items()
            if len(checked) == 0 and len(self.items) > 0:
                checked = [self.current]
            result = [item.value for item in checked]
        else:
            result = None if self.current is None else self.current.value
        self.result = result
        self.done = True
        self._changed()
        return result

    def cancel(self):
        """Close the menu with no selection; ``result`` is None."""
        if self.done:
            return self.result
        self.result = None
        self.done = True
        self._changed()
        return None


def run_menu(session, menu):
    """Standard key handling for a Menu; returns the menu's accept/cancel result.

    Up/Down move the cursor, Space toggles a checkbox (multi-select menus),
    Enter accepts, Escape cancels. Key releases and unknown keys are ignored.
    """
    while True:
        event = session.get()
        if not isinstance(event, KeyPress):
            continue
        if event.key == 'Up':
            menu.move(-1)
        elif event.key == 'Down':
            menu.move(1)
        elif event.key == 'Space':
            menu.toggle()
        elif event.key in ('Enter', 'Return'):
            return menu.accept()
        elif event.key == 'Escape':
            return menu.cancel()


class MenuPainter(CharGridPainter):
    """Paints a Menu: title, item rows (with checkboxes when multi-select),
    highlighted cursor row, key hints."""

    def __init__(self, scene, menu, **kwds):
        self._hint = ("Up/Down move  Space toggle  Enter accept  Esc cancel"
                      if menu.multi_select else
                      "Up/Down move  Enter select  Esc cancel")
        CharGridPainter.__init__(self, scene, menu, **kwds)

    def _item_text(self, item):
        if self.model.multi_select:
            return "[x] %s" % item.label if item.checked else "[ ] %s" % item.label
        return item.label

    def _content_shape(self):
        menu = self.model
        widths = [len(menu.title), len(self._hint)]
        widths += [len(self._item_text(item)) for item in menu.items]
        # title, blank, items, blank, hint
        return len(menu.items) + 4, min(max(widths), 100)

    def _render_content(self):
        menu = self.model
        self._write_line(0, menu.title, fg=self.TITLE_FG)
        for i, item in enumerate(menu.items):
            row = 2 + i
            if i == menu.cursor:
                self._fill_row(row, self.CURSOR_FG, self.CURSOR_BG)
            self._write_line(row, self._item_text(item))
        self._write_line(self.nrows - 1, self._hint, fg=self.HINT_FG)
