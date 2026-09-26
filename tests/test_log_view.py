"""Console scrolling, mouse routing to it, and the message-log dialog (headless)."""
import os

import pytest

from carriage_return.dialogs import LogDialog, open_log
from carriage_return.hud import WHEEL_ROWS, build_hud
from carriage_return.input import (GameplayInputHandler, InputDispatcher, KeyPress,
                                   MouseClick, MouseWheel)
from carriage_return.scene import Scene

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOIN_TIMEOUT = 10.0


def cell_text(grid, row):
    return ''.join(grid.registry.chars[i] for i in grid.glyph[row]).rstrip()


def visible(widget):
    """The widget's text rows, top to bottom."""
    return [''.join(widget.glyph[r]).rstrip() for r in range(widget.nrows)]


@pytest.fixture
def scene():
    os.chdir(PROJECT_ROOT)  # level1.png is loaded from cwd
    return Scene()


@pytest.fixture
def dispatcher():
    InputDispatcher.reset()
    yield InputDispatcher()
    InputDispatcher.reset()


def filled_scene(scene, n=40):
    for i in range(n):
        scene.write("line %d" % i)


def test_console_scrolls_back_and_clamps(scene):
    console = build_hud(scene).console
    filled_scene(scene)
    rows = console.nrows
    assert visible(console)[-1] == "line 39"

    console.scroll_by(5)
    assert visible(console)[-1] == "line 34"
    assert visible(console)[0] == "line %d" % (34 - rows + 1)

    console.scroll_by(1000)  # past the oldest line
    assert visible(console)[0] == "line 0"
    console.scroll_by(-1000)  # past the newest
    assert visible(console)[-1] == "line 39"
    assert console.scroll == 0


def test_console_accumulates_fractional_wheel_steps(scene):
    console = build_hud(scene).console
    filled_scene(scene)
    for _ in range(8):
        console.scroll_wheel(0.125)  # a precise trackpad's notch fraction
    assert console.scroll == pytest.approx(8 * 0.125 * WHEEL_ROWS)
    assert visible(console)[-1] == "line 36"


def test_new_entry_returns_view_to_newest(scene):
    console = build_hud(scene).console
    filled_scene(scene)
    console.scroll_by(10)
    scene.write("fresh")
    assert console.scroll == 0
    assert visible(console)[-1] == "fresh"


def test_advancing_the_turn_keeps_the_scroll(scene):
    console = build_hud(scene).console
    filled_scene(scene)
    console.scroll_by(10)
    scene.log.advance_turn()
    assert console.scroll == 10


def test_scroll_bar_thumb_tracks_position(scene):
    scene.screen.set_shape((40, 100))
    dialog_view = LogDialog(scene).view
    filled_scene(scene, 100)
    rows = dialog_view.nrows
    top, length = dialog_view._thumb
    assert top + length == rows  # following the tail: thumb at the bottom
    dialog_view.scroll_by(10 ** 6)
    assert dialog_view._thumb[0] == 0  # oldest: thumb at the top


def test_scroll_bar_click_pages_toward_the_click(scene):
    scene.screen.set_shape((40, 100))
    dialog = LogDialog(scene)
    filled_scene(scene, 100)
    view = dialog.view
    col = view.col_offset + view.ncols - 1
    view.click_scroll_bar(view.row_offset, col)  # track above the thumb
    assert view.scroll == view.nrows - 1
    view.click_scroll_bar(view.row_offset + view.nrows - 1, col)  # below it
    assert view.scroll == 0
    view.click_scroll_bar(view.row_offset, col - 5)  # not on the bar
    assert view.scroll == 0


def test_log_dialog_covers_most_of_the_screen_and_follows_resizes(scene):
    scene.screen.set_shape((40, 100))
    dialog = LogDialog(scene)
    rows, cols = dialog.layer.grid.shape
    assert rows >= 40 - 6 and cols >= 100 - 10
    scene.screen.set_shape((30, 80))
    assert dialog.layer.grid.shape[0] < rows
    dialog.close()
    assert scene.screen.changed.callbacks == []
    assert scene.log.changed.callbacks == []
    assert len(scene.grids) == 0


def test_log_session_scrolls_and_closes(scene, dispatcher):
    scene.screen.set_shape((40, 100))
    filled_scene(scene, 100)
    session = open_log(scene)
    grid = [g for g in scene.grids][0]
    assert cell_text(grid, 1).startswith("│Message log")

    session.post(MouseWheel(None, None, None, 2))
    session.post(KeyPress('PageUp'))
    session.post(KeyPress('Down'))
    session.post(KeyPress('Escape'))
    session.join(JOIN_TIMEOUT)
    assert session.finished.is_set()
    assert len(scene.grids) == 0
    assert not session.active


def test_gameplay_claims_only_mouse_events_over_the_console(scene):
    hud = build_hud(scene)
    handler = GameplayInputHandler(None, None, hud=hud, start_thread=False)
    c = hud.console
    over = MouseWheel(hud.layer.grid, c.row_offset + 1, c.col_offset + 1, 1)
    on_border = MouseWheel(hud.layer.grid, c.row_offset - 1, c.col_offset + 1, 1)
    over_map = MouseWheel(None, None, None, 1)
    assert handler.handle(over) is True
    assert handler.handle(on_border) is False
    assert handler.handle(over_map) is False
    assert handler.queue.qsize() == 1


def test_gameplay_wheel_scrolls_and_left_click_opens_log(scene):
    hud = build_hud(scene)
    filled_scene(scene)
    opened = []

    class Interp(object):
        def log(self, args):
            opened.append(args)

    handler = GameplayInputHandler(None, None, interpreter=Interp(), hud=hud,
                                   start_thread=False)
    grid = hud.layer.grid
    handler._process(MouseWheel(grid, 0, 0, 1))
    assert hud.console.scroll == WHEEL_ROWS
    handler._process(MouseClick(grid, 0, 0, 'right'))
    assert opened == []
    handler._process(MouseClick(grid, 0, 0, 'left'))
    assert opened == [[]]
