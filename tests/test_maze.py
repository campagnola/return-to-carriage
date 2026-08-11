"""Maze background colour: blocktype base colour plus layered washes."""
import numpy as np

from carriage_return.blocktypes import BlockTypes
from carriage_return.maze import Maze


def void_maze(shape=(4, 4)):
    """A maze of 'void' cells: the one blocktype with bg_color_var=0, so its
    bg_color is deterministic and washes can be checked exactly.
    """
    bt = BlockTypes()
    maze = Maze.filled(shape, bt, 'void')
    base = bt['bg_color'][bt.id_of('void')][:3].copy()
    return maze, base


def test_wash_bg_color_single_layer_matches_previous_behavior():
    maze, base = void_maze()
    rgb = np.zeros(maze.shape + (3,), dtype='float32')
    rgb[..., 0] = 1.0
    amount = 0.4
    maze.wash_bg_color(rgb, amount)

    expected = base * (1 - amount) + rgb[0, 0] * amount
    assert np.allclose(maze.bg_color[..., :3], expected)


def test_wash_bg_color_layers_in_order_on_overlap():
    """Two washes covering the same cells compose sequentially, not by
    replacement: applying wash A then wash B gives a different (and correct)
    result from B alone, in the order the calls were made.
    """
    maze, base = void_maze()
    mask = np.ones(maze.shape, dtype=bool)

    rgb_a = np.full(maze.shape + (3,), (1.0, 0.0, 0.0), dtype='float32')
    amount_a = 0.5
    rgb_b = np.full(maze.shape + (3,), (0.0, 1.0, 0.0), dtype='float32')
    amount_b = 0.5

    maze.wash_bg_color(rgb_a, amount_a, mask=mask)
    maze.wash_bg_color(rgb_b, amount_b, mask=mask)

    after_a = base * (1 - amount_a) + rgb_a[0, 0] * amount_a
    expected = after_a * (1 - amount_b) + rgb_b[0, 0] * amount_b
    assert np.allclose(maze.bg_color[..., :3], expected)
    # Sanity: this is not the same as B alone clobbering A (the old,
    # single-slot behavior).
    assert not np.allclose(maze.bg_color[..., :3], base * (1 - amount_b) + rgb_b[0, 0] * amount_b)


def test_wash_bg_color_disjoint_masks_stay_independent():
    maze, base = void_maze(shape=(2, 4))
    left_mask = np.zeros(maze.shape, dtype=bool)
    left_mask[:, :2] = True
    right_mask = ~left_mask

    rgb_a = np.full(maze.shape + (3,), (1.0, 0.0, 0.0), dtype='float32')
    rgb_b = np.full(maze.shape + (3,), (0.0, 0.0, 1.0), dtype='float32')
    maze.wash_bg_color(rgb_a, 0.5, mask=left_mask)
    maze.wash_bg_color(rgb_b, 0.5, mask=right_mask)

    bg = maze.bg_color
    expected_left = base * 0.5 + rgb_a[0, 0] * 0.5
    expected_right = base * 0.5 + rgb_b[0, 0] * 0.5
    assert np.allclose(bg[left_mask][:, :3], expected_left)
    assert np.allclose(bg[right_mask][:, :3], expected_right)


def test_invalidate_appearance_keeps_washes_by_default():
    maze, base = void_maze()
    rgb = np.full(maze.shape + (3,), (1.0, 0.0, 0.0), dtype='float32')
    maze.wash_bg_color(rgb, 0.5)
    washed = maze.bg_color[..., :3].copy()

    maze.invalidate_appearance()  # e.g. after an unrelated blocks edit
    assert np.allclose(maze.bg_color[..., :3], washed)


def test_invalidate_appearance_can_clear_washes():
    maze, base = void_maze()
    rgb = np.full(maze.shape + (3,), (1.0, 0.0, 0.0), dtype='float32')
    maze.wash_bg_color(rgb, 0.5)
    assert not np.allclose(maze.bg_color[..., :3], base)

    maze.invalidate_appearance(clear_washes=True)
    assert np.allclose(maze.bg_color[..., :3], base)
