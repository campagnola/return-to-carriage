"""The Menu dialog: interactive state, key loop, and drawing in one widget.

``MenuItem`` is a pure-data entry (a label, a value, a checked flag).
``MenuWidget`` is a ``carriage_return.widgets.Widget`` that holds the
interactive state (title, items, cursor, done/result) and draws itself;
``run_menu`` is the sequential key loop a DialogSession runs on its own
thread. See ``dialogs/__init__.py`` for how a MenuWidget is wrapped in a
bordered ``GridFrame`` and composited to the screen.
"""
from ..input import KeyPress
from ..widgets import CURSOR_BG, CURSOR_FG, FG, BG, HINT_FG, TITLE_FG, Widget


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


class MenuWidget(Widget):
    """A list of selectable items with a cursor, optionally with checkboxes.

    Items may be given as MenuItem instances or plain strings (auto-wrapped).
    In multi_select mode toggle() checks/unchecks the item under the cursor
    and accept() returns the list of checked items' values (falling back to
    the item under the cursor when nothing is checked). In single-select mode
    accept() returns the current item's value.

    accept() and cancel() set ``done`` (and ``result``) so a caller can
    observe closure; they are idempotent once done. Every mutator repaints
    the widget's own cells before returning, so a caller never needs to
    trigger a separate render pass.
    """

    def __init__(self, title, items, multi_select=False):
        Widget.__init__(self)
        self.title = title
        self.items = [item if isinstance(item, MenuItem) else MenuItem(item)
                      for item in items]
        self.multi_select = multi_select
        self.cursor = 0
        self.done = False
        self.result = None
        self._hint = ("Up/Down move  Space toggle  Enter accept  Esc cancel"
                      if multi_select else
                      "Up/Down move  Enter select  Esc cancel")

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
            self.repaint()

    def toggle(self):
        """Toggle the checkbox under the cursor (no-op unless multi_select)."""
        if not self.multi_select or len(self.items) == 0:
            return
        item = self.items[self.cursor]
        item.checked = not item.checked
        self._changed()
        self.repaint()

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
        self.repaint()
        return result

    def cancel(self):
        """Close the menu with no selection; ``result`` is None."""
        if self.done:
            return self.result
        self.result = None
        self.done = True
        self._changed()
        self.repaint()
        return None

    # -- sizing/drawing ------------------------------------------------

    def _item_text(self, item):
        if self.multi_select:
            return "[x] %s" % item.label if item.checked else "[ ] %s" % item.label
        return item.label

    def preferred_shape(self):
        """(nrows, ncols) this menu wants: title/blank/items/blank/hint, capped at 100 cols."""
        widths = [len(self.title), len(self._hint)]
        widths += [len(self._item_text(item)) for item in self.items]
        return len(self.items) + 4, min(max(widths), 100)

    def _shape_changed(self):
        self.repaint()

    def repaint(self):
        """Redraw the whole widget from current state (title, items, cursor, hint)."""
        self.clear(fg=FG, bg=BG)
        self.write(0, 0, self.title, fg=TITLE_FG)
        for i, item in enumerate(self.items):
            row = 2 + i
            if i == self.cursor:
                self.fill_row(row, fg=CURSOR_FG, bg=CURSOR_BG)
            self.write(row, 0, self._item_text(item))
        self.write(self.nrows - 1, 0, self._hint, fg=HINT_FG)


def run_menu(session, menu):
    """Standard key handling for a MenuWidget; returns the menu's accept/cancel result.

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
