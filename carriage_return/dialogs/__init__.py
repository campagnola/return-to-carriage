"""Game-side dialog subpackage: widgets, key loops, and the
open_menu/open_pager/open_cast lifecycle helpers.

A dialog is entirely game state: its display is a screen-space CharGridLayer
in ``scene.grids`` (composited from a ``carriage_return.widgets`` tree by a
``WidgetGridLayer``) and its input is the dispatcher stack (the
DialogSession is itself the top handler). ``open_menu``/``open_pager``/
``open_cast`` own the whole lifecycle -- build the content widget, wrap it
in a bordered ``GridFrame``, wrap that in a ``WidgetGridLayer``, push the
session, arrange teardown, and start the dialog thread -- so opening a
dialog needs no UI and works headless by construction.

Callers get results via ``session.finished.connect(cb)`` (called on the
dialog thread with the session); ``session.result`` is None on cancel.
"""
from ..widgets import GridFrame, WidgetGridLayer
from .cast import CastWidget, run_cast
from .menu import MenuItem, MenuWidget, run_menu
from .pager import PagerWidget, run_pager
from .session import DialogClosed, DialogSession

__all__ = [
    'open_menu', 'open_pager', 'open_cast',
    'DialogSession', 'DialogClosed',
    'MenuWidget', 'MenuItem', 'PagerWidget', 'CastWidget',
    'run_menu', 'run_pager', 'run_cast',
]


def _as_menu_items(items):
    """Coerce *items* into things MenuWidget accepts.

    MenuItem and plain strings pass through (MenuWidget wraps strings
    itself). Any other object is treated as a game item and wrapped as
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


def _wrap_dialog(scene, content):
    """Size a 1x1 GridFrame around *content* to its preferred_shape(), paint
    it, and composite the tree into its own centered WidgetGridLayer."""
    nrows, ncols = content.preferred_shape()
    frame = GridFrame([nrows], [ncols], [(content, 0, 0, 1, 1)])
    # GridFrame's initial layout pass places the child via add_child(), which
    # doesn't invoke its _shape_changed() hook -- paint its starting content
    # explicitly now that it has real cells (same as hud.py's Hud.__init__).
    content.repaint()
    return WidgetGridLayer(scene, frame, anchor='center')


def open_menu(scene, title, items, multi_select=False):
    """Open a modal menu over the game and return its (started) session.

    *items* may be MenuItem instances, plain strings, or game items (wrapped
    as ``MenuItem(item.description, value=item)``). The menu is drawn into a
    screen-space grid and the session captures all input until the body
    returns; teardown (pop the session, remove the grid) is pure game state
    and runs on the dialog thread. The renderer notices the grid disappeared
    at its next frame tick.
    """
    menu = MenuWidget(title, _as_menu_items(items), multi_select)
    layer = _wrap_dialog(scene, menu)
    return _run_dialog(lambda s: run_menu(s, menu), layer, title)


def open_pager(scene, title, pages):
    """Open a modal pager (multi-page text) and return its (started) session.

    Mirrors open_menu: the pages are drawn into a screen-space grid, the
    session captures input, and teardown runs on the dialog thread.
    """
    pager = PagerWidget(title, pages)
    layer = _wrap_dialog(scene, pager)
    return _run_dialog(lambda s: run_pager(s, pager), layer, title)


def open_cast(scene, spells):
    """Open the modal spell-casting prompt and return its (started) session.

    *spells* is the ``{name: factory}`` registry the typed name is matched
    against (see ``spell.SPELLS``). The session captures all input while open;
    its result is ``(spell_name, arrow_key)`` on cast, or None if cancelled.
    The caller (the interpreter) turns that into a spell in the world.
    """
    prompt = CastWidget(spells)
    layer = _wrap_dialog(scene, prompt)
    return _run_dialog(lambda s: run_cast(s, prompt), layer, "cast")


def _run_dialog(body, layer, name):
    """Wire a dialog body to its WidgetGridLayer and start it.

    Pushes the session onto the dispatcher stack, arranges teardown (drop the
    session, close the layer) on completion, and starts the dialog thread.
    """
    session = DialogSession(body, name=name)
    session.activate()

    def teardown(_):
        # dialog thread; game state only -- nothing here touches the GUI
        if session.active:
            session.deactivate()
        layer.close()

    session.finished.connect(teardown)
    return session.start()
