"""Game-side dialog subpackage: models, key loops, painters, and the
open_menu/open_pager lifecycle helpers.

A dialog is entirely game state: its display is a screen-space CharGridLayer
in ``scene.grids`` (written by a painter) and its input is the dispatcher
stack (the DialogSession is itself the top handler). ``open_menu`` and
``open_pager`` own the whole lifecycle — build the model + painter, push the
session, arrange teardown, and start the dialog thread — so opening a dialog
needs no UI and works headless by construction.

Callers get results via ``session.finished.connect(cb)`` (called on the
dialog thread with the session); ``session.result`` is None on cancel.
"""
from .base import CharGridPainter, Widget
from .menu import Menu, MenuItem, MenuPainter, run_menu
from .pager import Pager, PagerPainter, run_pager
from .session import DialogClosed, DialogSession

__all__ = [
    'open_menu', 'open_pager',
    'DialogSession', 'DialogClosed',
    'Menu', 'MenuItem', 'Pager',
    'run_menu', 'run_pager',
    'MenuPainter', 'PagerPainter',
    'CharGridPainter', 'Widget',
]


def _as_menu_items(items):
    """Coerce *items* into things Menu accepts.

    MenuItem and plain strings pass through (Menu wraps strings itself). Any
    other object is treated as a game item and wrapped as
    ``MenuItem(item.description, value=item)`` so the selection returns the
    item, not its label.
    """
    coerced = []
    for item in items:
        if isinstance(item, (MenuItem, str)):
            coerced.append(item)
        else:
            coerced.append(MenuItem(item.description, value=item))
    return coerced


def open_menu(scene, title, items, multi_select=False):
    """Open a modal menu over the game and return its (started) session.

    *items* may be MenuItem instances, plain strings, or game items (wrapped
    as ``MenuItem(item.description, value=item)``). A MenuPainter writes the
    menu into a screen-space grid and the session captures all input until
    the body returns; teardown (pop the session, remove the grid) is pure
    game state and runs on the dialog thread. The renderer notices the grid
    disappeared at its next frame tick.
    """
    menu = Menu(title, _as_menu_items(items), multi_select)
    painter = MenuPainter(scene, menu)
    return _run_dialog(lambda s: run_menu(s, menu), painter, title)


def open_pager(scene, title, pages):
    """Open a modal pager (multi-page text) and return its (started) session.

    Mirrors open_menu: a PagerPainter writes the pages into a screen-space
    grid, the session captures input, and teardown runs on the dialog thread.
    """
    pager = Pager(title, pages)
    painter = PagerPainter(scene, pager)
    return _run_dialog(lambda s: run_pager(s, pager), painter, title)


def _run_dialog(body, painter, name):
    """Wire a dialog body to its painter's grid and start it.

    Pushes the session onto the dispatcher stack, arranges teardown (drop the
    session, close the painter) on completion, and starts the dialog thread.
    """
    session = DialogSession(body, name=name)
    session.activate()

    def teardown(_):
        # dialog thread; game state only — nothing here touches the GUI
        if session.active:
            session.deactivate()
        painter.close()

    session.finished.connect(teardown)
    return session.start()
