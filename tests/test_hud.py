"""The HUD widget tree renders game state into one screen grid (fully headless)."""
import os

import pytest

from carriage_return.hud import build_hud, Hud, STATS_TEXT
from carriage_return.scene import Scene

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def row_text(grid, row):
    return ''.join(grid.registry.chars[i] for i in grid.glyph[row])


def widget_text(widget, row):
    return ''.join(widget.glyph[row]).rstrip()


@pytest.fixture
def scene():
    os.chdir(PROJECT_ROOT)  # level1.png is loaded from cwd
    return Scene()


def test_build_hud_creates_one_screen_grid(scene):
    hud = build_hud(scene)
    assert list(scene.grids) == [hud.layer.grid]
    assert hud.layer.grid.space == 'screen'
    assert hud.layer.grid.anchor == 'bottom-left'
    assert hud.layer.grid.shape == hud.frame.shape


def test_stats_text_reaches_the_grid(scene):
    hud = build_hud(scene)
    width = hud.stats.ncols
    assert widget_text(hud.stats, 0) == STATS_TEXT[:width].rstrip()
    # composited into the grid inside the outer border
    assert row_text(hud.layer.grid, 1)[1:width + 1].rstrip() == STATS_TEXT[:width].rstrip()


def check_borders(hud):
    """Outer ring, the stats divider, and the info/console divider share junctions."""
    grid = hud.layer.grid
    rows, cols = grid.shape
    split = hud.console.col_offset - 1      # column of the info/console divider
    divider = hud.info.row_offset - 1       # row under the stats bar

    top, mid, bottom = row_text(grid, 0), row_text(grid, divider), row_text(grid, rows - 1)
    assert top == '┌' + '─' * (cols - 2) + '┐'
    assert mid == '├' + '─' * (split - 1) + '┬' + '─' * (cols - split - 2) + '┤'
    assert bottom == '└' + '─' * (split - 1) + '┴' + '─' * (cols - split - 2) + '┘'
    for r in range(divider + 1, rows - 1):
        line = row_text(grid, r)
        assert line[0] == line[split] == line[-1] == '│'


def test_frame_draws_shared_borders(scene):
    check_borders(build_hud(scene))


def test_console_renders_log_tail_newest_at_bottom(scene):
    hud = build_hud(scene)
    console = hud.console
    rows = console.nrows

    scene.write("first")
    scene.write("second")
    assert widget_text(console, rows - 2) == "first"
    assert widget_text(console, rows - 1) == "second"

    # overflow: only the tail stays visible
    for i in range(rows + 5):
        scene.write("line %d" % i)
    assert widget_text(console, 0) == "line %d" % (5)
    assert widget_text(console, rows - 1) == "line %d" % (rows + 4)


def test_console_wraps_long_lines(scene):
    hud = build_hud(scene)
    console = hud.console
    rows, cols = console.shape
    scene.write("x" * (cols + 5))
    assert widget_text(console, rows - 2) == "x" * cols
    assert widget_text(console, rows - 1) == "x" * 5


def test_hud_relays_out_and_keeps_content_on_screen_change(scene):
    hud = build_hud(scene)
    scene.write("hello")
    hud.info.set_text("abcdefghij" * 10)
    scene.screen.set_shape((30, 60))

    grid = hud.layer.grid
    assert grid.shape == hud.frame.shape == (Hud.stats_rows + Hud.box_rows - 1, 60)
    assert hud.stats.nrows == Hud.stats_rows - 2
    assert hud.info.nrows == hud.console.nrows == Hud.box_rows - 2
    # info + console + three border columns fill the width exactly
    assert hud.info.ncols + hud.console.ncols + 3 == 60

    # children repainted into the new layout rather than wiped by the frame
    assert widget_text(hud.stats, 0) == STATS_TEXT[:hud.stats.ncols].rstrip()
    assert widget_text(hud.console, hud.console.nrows - 1) == "hello"
    info_w = hud.info.ncols
    text = "abcdefghij" * 10
    assert widget_text(hud.info, 0) == text[:info_w]
    assert widget_text(hud.info, 1) == text[info_w:2 * info_w]
    assert "hello" in row_text(grid, grid.shape[0] - 2)

    check_borders(hud)


def test_console_repaint_is_observer_driven(scene):
    hud = build_hud(scene)
    grid = hud.layer.grid
    version = grid.version
    scene.write("ping")
    assert grid.version > version  # log observer repainted the grid
    assert "ping" in row_text(grid, grid.shape[0] - 2)


def test_hud_close_removes_grids_and_observers(scene):
    hud = build_hud(scene)
    hud.close()
    assert len(scene.grids) == 0
    assert scene.log.changed.callbacks == []
    assert scene.screen.changed.callbacks == []
