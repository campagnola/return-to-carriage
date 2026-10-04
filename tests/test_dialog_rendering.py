"""Dialog widgets render, framed, into scene.grids and follow their own changes (headless).

Uses dialogs._wrap_dialog, the same framing open_menu/open_pager/open_cast
apply, without starting a dialog thread.
"""
import os

import numpy as np
import pytest

from carriage_return.dialogs import MenuWidget, PagerWidget, _wrap_dialog
from carriage_return.scene import Scene
from carriage_return.widgets import CURSOR_BG

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def row_text(grid, row):
    return ''.join(grid.registry.chars[i] for i in grid.glyph[row])


def grid_text(grid):
    return [row_text(grid, r) for r in range(grid.shape[0])]


@pytest.fixture
def scene():
    os.chdir(PROJECT_ROOT)
    return Scene()


def test_menu_layout(scene):
    menu = MenuWidget("Take which items?", ["a scroll", "a torch"], multi_select=True)
    layer = _wrap_dialog(scene, menu)
    grid = layer.grid
    assert list(scene.grids) == [grid] and grid.space == 'screen'

    rows = grid_text(grid)
    width = grid.shape[1]
    assert rows[0] == '┌' + '─' * (width - 2) + '┐'
    assert rows[-1] == '└' + '─' * (width - 2) + '┘'
    assert all(r[0] == r[-1] == '│' for r in rows[1:-1])
    # title, blank, items with checkboxes, blank, hint -- inside the border
    assert rows[1][1:].startswith("Take which items?")
    assert rows[3][1:].startswith("[ ] a scroll")
    assert rows[4][1:].startswith("[ ] a torch")
    assert "Esc cancel" in rows[-2]


def test_menu_grid_follows_widget_changes(scene):
    menu = MenuWidget("Pick", ["one", "two"], multi_select=True)
    grid = _wrap_dialog(scene, menu).grid
    cursor_bg = np.float32(CURSOR_BG)

    def item_bg(i):
        return grid.bgcolor[3 + i, 1:-1]   # border row + title + blank, inside the side borders

    assert (item_bg(0) == cursor_bg).all()
    assert not (item_bg(1) == cursor_bg).all()
    version = grid.version
    menu.move(1)
    assert grid.version > version  # observer repainted the grid
    assert (item_bg(1) == cursor_bg).all()
    assert not (item_bg(0) == cursor_bg).all()

    menu.toggle()
    assert "[x] two" in grid_text(grid)[4]


def test_pager_layout_and_paging(scene):
    pager = PagerWidget("a scroll", ["page one text", "page two text"])
    grid = _wrap_dialog(scene, pager).grid
    rows = grid_text(grid)
    assert rows[1][1:].startswith("a scroll")
    assert rows[3][1:].startswith("page one text")
    assert "page 1/2" in rows[-2]

    pager.next_page()
    rows = grid_text(grid)
    assert rows[3][1:].startswith("page two text")
    assert "page 2/2" in rows[-2]


def test_close_detaches(scene):
    menu = MenuWidget("Pick", ["one", "two"])
    layer = _wrap_dialog(scene, menu)
    assert len(scene.grids) == 1
    layer.close()
    assert len(scene.grids) == 0
    # further widget changes must not touch the removed grid
    version = layer.grid.version
    menu.move(1)
    assert layer.grid.version == version
