"""Dialog painters render widget models into scene.grids (fully headless)."""
import os

import pytest

from carriage_return.dialogs import MenuPainter, PagerPainter, Menu, Pager
from carriage_return.scene import Scene

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def row_text(grid, row):
    return ''.join(grid.registry.chars[i] for i in grid.glyph[row])


def grid_text(grid):
    return [row_text(grid, r) for r in range(grid.shape[0])]


@pytest.fixture
def scene():
    os.chdir(PROJECT_ROOT)
    return Scene()


def test_menu_painter_layout(scene):
    menu = Menu("Take which items?", ["a scroll", "a torch"], multi_select=True)
    painter = MenuPainter(scene, menu)
    grid = painter.grid
    assert grid in list(scene.grids) and grid.space == 'screen'

    rows = grid_text(grid)
    # border drawn as characters
    assert rows[0].startswith('+') and rows[0].endswith('+') and '-' in rows[0]
    assert rows[-1].startswith('+') and rows[-1].endswith('+')
    assert all(r.startswith('|') and r.endswith('|') for r in rows[1:-1])
    # title, items with checkboxes, hint line
    assert "Take which items?" in rows[2]
    assert "[ ] a scroll" in rows[4]
    assert "[ ] a torch" in rows[5]
    assert "Esc cancel" in rows[-3]


def test_menu_painter_follows_model_changes(scene):
    menu = Menu("Pick", ["one", "two"], multi_select=True)
    painter = MenuPainter(scene, menu)
    grid = painter.grid

    # cursor bar on the row under the cursor
    import numpy as np
    cursor_bg = np.float32(painter.CURSOR_BG)

    def cursor_row_colors(i):
        return grid.bgcolor[2 + painter.padding + 1 + i]

    assert (cursor_row_colors(0)[2] == cursor_bg).all()
    version = grid.version
    menu.move(1)
    assert grid.version > version  # observer repainted
    assert (cursor_row_colors(1)[2] == cursor_bg).all()

    menu.toggle()
    assert "[x] two" in grid_text(grid)[5]


def test_pager_painter_layout_and_paging(scene):
    pager = Pager("a scroll", ["page one text", "page two text"])
    painter = PagerPainter(scene, pager)
    rows = grid_text(painter.grid)
    assert "a scroll" in rows[2]
    assert "page one text" in '\n'.join(rows)
    assert "page 1/2" in rows[-3]

    pager.next_page()
    rows = grid_text(painter.grid)
    assert "page two text" in '\n'.join(rows)
    assert "page 2/2" in rows[-3]


def test_painter_close_detaches(scene):
    menu = Menu("Pick", ["one"])
    painter = MenuPainter(scene, menu)
    assert len(scene.grids) == 1
    painter.close()
    assert len(scene.grids) == 0
    assert menu.observer is None
    # further model changes must not touch the removed grid
    version = painter.grid.version
    menu.move(0)
    assert painter.grid.version == version
