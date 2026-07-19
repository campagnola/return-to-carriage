"""HUD painters render game state into scene.grids (fully headless)."""
import os

import pytest

from carriage_return.hud import build_hud, Hud, STATS_TEXT, INFO_TEXT
from carriage_return.scene import Scene

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def row_text(grid, row):
    return ''.join(grid.registry.chars[i] for i in grid.glyph[row]).rstrip()


def content_text(grid, row):
    """Text of *row* inside the one-cell border ring."""
    return ''.join(grid.registry.chars[i] for i in grid.glyph[row, 1:-1]).rstrip()


@pytest.fixture
def scene():
    os.chdir(PROJECT_ROOT)  # level1.png is loaded from cwd
    return Scene()


def test_build_hud_creates_three_screen_grids(scene):
    hud = build_hud(scene)
    grids = list(scene.grids)
    assert len(grids) == 3
    assert {g.space for g in grids} == {'screen'}
    assert hud.console.grid in grids
    assert hud.stats.grid in grids
    assert hud.info.grid in grids


def test_static_painters_render_their_text(scene):
    hud = build_hud(scene)
    assert content_text(hud.stats.grid, 1) == STATS_TEXT[:hud.stats.grid.shape[1] - 2].rstrip()
    assert content_text(hud.info.grid, 1) == INFO_TEXT


def test_painters_draw_borders(scene):
    hud = build_hud(scene)
    for grid in (hud.console.grid, hud.stats.grid, hud.info.grid):
        rows, cols = grid.shape
        bar = '+' + '-' * (cols - 2) + '+'
        assert row_text(grid, 0) == bar
        assert row_text(grid, rows - 1) == bar
        assert row_text(grid, 1).startswith('|') and row_text(grid, 1).endswith('|')


def test_console_renders_log_tail_newest_at_bottom(scene):
    hud = build_hud(scene)
    grid = hud.console.grid
    rows = grid.shape[0]
    content_rows = rows - 2  # inside the border ring

    scene.write("first")
    scene.write("second")
    assert content_text(grid, rows - 3) == "first"
    assert content_text(grid, rows - 2) == "second"

    # overflow: only the tail stays visible
    for i in range(rows + 5):
        scene.write("line %d" % i)
    n_lines = 2 + rows + 5
    assert content_text(grid, 1) == "line %d" % (n_lines - content_rows - 2)
    assert content_text(grid, rows - 2) == "line %d" % (rows + 4)


def test_console_wraps_long_lines(scene):
    hud = build_hud(scene)
    grid = hud.console.grid
    rows, cols = grid.shape
    w = cols - 2
    scene.write("x" * (w + 5))
    assert content_text(grid, rows - 3) == "x" * w
    assert content_text(grid, rows - 2) == "x" * 5


def test_hud_reshapes_and_rewraps_on_screen_change(scene):
    hud = build_hud(scene)
    scene.screen.set_shape((30, 60))

    assert hud.stats.grid.shape == (Hud.stats_rows, 60)
    assert hud.info.grid.shape[0] == hud.console.grid.shape[0] == Hud.box_rows
    # info + console split the width exactly: no overlap, no gap
    assert hud.info.grid.shape[1] + hud.console.grid.shape[1] == 60

    # text re-wrapped into the narrower info box
    info_w = hud.info.grid.shape[1] - 2
    assert content_text(hud.info.grid, 1) == INFO_TEXT[:info_w]
    assert content_text(hud.info.grid, 2) == INFO_TEXT[info_w:2 * info_w]

    # borders redrawn on the new outline
    for grid in (hud.console.grid, hud.stats.grid, hud.info.grid):
        bar = '+' + '-' * (grid.shape[1] - 2) + '+'
        assert row_text(grid, 0) == bar
        assert row_text(grid, grid.shape[0] - 1) == bar


def test_console_repaint_is_observer_driven(scene):
    hud = build_hud(scene)
    grid = hud.console.grid
    version = grid.version
    scene.write("ping")
    assert grid.version > version  # log observer repainted the grid


def test_hud_close_removes_grids_and_observers(scene):
    hud = build_hud(scene)
    hud.close()
    assert len(scene.grids) == 0
    assert scene.log.changed.callbacks == []
    assert scene.screen.changed.callbacks == []
