"""Maze background colour (blocktype base colour plus layered washes), and the
shared opacity helpers: ``Maze.opaque``, ``neighbour_shifts`` and the
line-of-sight geometry mask built from them."""
import numpy as np

from carriage_return.blocktypes import BlockTypes
from carriage_return.light import ArrayLight
from carriage_return.maze import DIAGONAL_OFFSETS, SIDE_OFFSETS, Maze, neighbour_shifts


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


# --- Maze.opaque, neighbour_shifts, _opaque_geometry_mask -------------------

def wall_path_maze(pattern):
    """A maze from a list of strings, ``'#'`` for wall and ``'.'`` for path.

    Row ``i`` of *pattern* is ``blocks[i]`` (numpy row order, not flipped).
    """
    bt = BlockTypes()
    ids = {'#': bt.id_of('wall'), '.': bt.id_of('path')}
    blocks = np.array([[ids[c] for c in row] for row in pattern], dtype='uint8')
    return Maze(blocks, bt)


def test_opaque_is_opacity_above_half():
    maze = wall_path_maze(['#.',
                           '.#'])
    assert maze.opaque.dtype == bool
    assert np.array_equal(maze.opaque, maze.opacity > 0.5)
    assert np.array_equal(maze.opaque, [[True, False], [False, True]])


def test_opaque_is_cached_and_invalidated_with_blocks():
    maze = wall_path_maze(['#.',
                           '.#'])
    first = maze.opaque
    assert maze.opaque is first  # cached, like opacity

    maze.blocks[0, 1] = maze.blocktypes.id_of('wall')
    assert maze.opaque is first  # stale until invalidated, like opacity
    maze.invalidate_appearance()
    assert maze.opaque is not first
    assert np.array_equal(maze.opaque, [[True, True], [False, True]])


# A small, deliberately asymmetric mask so every direction gives a
# different answer:
#   row 0:  T F F
#   row 1:  F T F
#   row 2:  F F F
#   row 3:  T T F
ASYM = np.array([[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 0],
                 [1, 1, 0]], dtype=bool)


def reference_neighbour(mask, dr, dc, pad):
    """``mask[i + dr, j + dc]`` for every cell, *pad* where that is off-map."""
    rows, cols = mask.shape
    out = np.full(mask.shape, pad, dtype=bool)
    for i in range(rows):
        for j in range(cols):
            if 0 <= i + dr < rows and 0 <= j + dc < cols:
                out[i, j] = mask[i + dr, j + dc]
    return out


def test_neighbour_shifts_has_all_eight_directions():
    n = neighbour_shifts(ASYM, pad=False)
    assert set(n) == set(SIDE_OFFSETS) | set(DIAGONAL_OFFSETS)
    assert len(SIDE_OFFSETS) == len(DIAGONAL_OFFSETS) == 4


def test_neighbour_shifts_matches_reference_for_both_pads():
    for pad in (False, True):
        n = neighbour_shifts(ASYM, pad=pad)
        for (dr, dc), got in n.items():
            assert got.shape == ASYM.shape and got.dtype == bool
            assert np.array_equal(got, reference_neighbour(ASYM, dr, dc, pad)), (pad, dr, dc)


def test_neighbour_shifts_directions_are_row_col():
    """Spot-check the (row, col) convention by hand on ASYM, one per direction."""
    n = neighbour_shifts(ASYM, pad=False)
    # (-1, 0), one row before: [1, 0] sees [0, 0] (T); [1, 1] sees [0, 1] (F)
    assert n[-1, 0][1, 0] and not n[-1, 0][1, 1]
    # (1, 0), one row after: [2, 0] sees [3, 0] (T); [2, 2] sees [3, 2] (F)
    assert n[1, 0][2, 0] and not n[1, 0][2, 2]
    # (0, 1), one column after: [3, 0] sees [3, 1] (T); [3, 1] sees [3, 2] (F)
    assert n[0, 1][3, 0] and not n[0, 1][3, 1]
    # (0, -1), one column before: [1, 2] sees [1, 1] (T); [1, 1] sees [1, 0] (F)
    assert n[0, -1][1, 2] and not n[0, -1][1, 1]
    # (1, 1): [0, 0] sees [1, 1] (T); [1, 1] sees [2, 2] (F)
    assert n[1, 1][0, 0] and not n[1, 1][1, 1]
    # (-1, -1): [1, 1] sees [0, 0] (T); [3, 1] sees [2, 0] (F)
    assert n[-1, -1][1, 1] and not n[-1, -1][3, 1]
    # (1, -1): [2, 1] sees [3, 0] (T); [0, 1] sees [1, 0] (F)
    assert n[1, -1][2, 1] and not n[1, -1][0, 1]
    # (-1, 1): [1, 0] sees [0, 1] (F); [2, 0] sees [1, 1] (T)
    assert not n[-1, 1][1, 0] and n[-1, 1][2, 0]


def test_neighbour_shifts_pad_fills_only_the_off_map_ring():
    empty = np.zeros((3, 4), dtype=bool)
    n = neighbour_shifts(empty, pad=True)
    assert n[-1, 0][0].all() and not n[-1, 0][1:].any()
    assert n[0, 1][:, -1].all() and not n[0, 1][:, :-1].any()
    assert n[1, 1][-1].all() and n[1, 1][:, -1].all() and not n[1, 1][:-1, :-1].any()
    assert not any(v.any() for v in neighbour_shifts(empty, pad=False).values())


def reference_opaque_geometry_mask(opaque):
    """Straightforward per-cell rewrite of the 3x3 join rule."""
    rows, cols = opaque.shape

    def at(i, j):
        return 0 <= i < rows and 0 <= j < cols and opaque[i, j]

    out = np.zeros((rows * 3, cols * 3), dtype=bool)
    for i in range(rows):
        for j in range(cols):
            if not opaque[i, j]:
                continue
            block = out[3 * i:3 * i + 3, 3 * j:3 * j + 3]
            block[1, 1] = True
            for dr, dc in SIDE_OFFSETS:
                block[1 + dr, 1 + dc] = at(i + dr, j + dc)
            for dr, dc in DIAGONAL_OFFSETS:
                block[1 + dr, 1 + dc] = (at(i + dr, j + dc)
                                         and block[1 + dr, 1] and block[1, 1 + dc])
    return out


def test_opaque_geometry_mask_matches_reference():
    """Regression: the refactor onto neighbour_shifts leaves the mask unchanged."""
    rng = np.random.default_rng(1234)
    bt = BlockTypes()
    ids = np.array([bt.id_of('path'), bt.id_of('wall')], dtype='uint8')
    for shape in [(1, 1), (2, 5), (9, 7), (23, 31)]:
        maze = Maze(ids[rng.integers(0, 2, size=shape)], bt)
        got = maze._opaque_geometry_mask()
        assert got.dtype == bool and got.shape == (shape[0] * 3, shape[1] * 3)
        assert np.array_equal(got, reference_opaque_geometry_mask(maze.opaque)), shape


def test_opaque_geometry_mask_hand_checked():
    # An L of wall, plus a lone wall touching it only diagonally (no join).
    maze = wall_path_maze(['##.',
                           '#..',
                           '..#'])
    expected = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=bool)
    assert np.array_equal(maze._opaque_geometry_mask(), expected)


def test_array_light_upsamples_to_field():
    """Regression: ArrayLight's map is its array blown up to field resolution,
    times its colour."""
    maze = wall_path_maze(['..',
                           '..',
                           '..'])
    arr = np.arange(6, dtype='float32').reshape(3, 2)
    color = np.array([1, 2, 3], dtype='float32')
    light = maze.add_light(ArrayLight(maze, arr, color=tuple(color)), pos=(0, 0))
    ss = 4
    got = light.lightmap(supersample=ss)
    assert got.shape == (3 * ss, 2 * ss, 3)
    for i in range(3 * ss):
        for j in range(2 * ss):
            assert np.array_equal(got[i, j], arr[i // ss, j // ss] * color)
