"""WaterAnimation: the river's flowing noise field and its display modifier.

Headless and synchronous, like test_heat: WaterAnimation is built with
``start=False`` so no animation thread spawns, and driven a tick at a time
through ``advance()``.
"""
import numpy as np
import pytest

from carriage_return.blocktypes import BlockTypes
from carriage_return.layers import GlyphRegistry, SpriteLayer
from carriage_return.levels import level_001_home
from carriage_return.maze import Maze
from carriage_return.terrain.water import WATER_COLOR_RAMP_RGB, WaterAnimation, WaterBody
from carriage_return.world import Level


def water_room(rows=5, cols=3):
    """A room that is entirely river, flowing straight downstream (+y)."""
    bt = BlockTypes()
    maze = Maze.filled((rows, cols), bt, 'grass')
    Level('room', maze)
    river_id = bt.id_of('river')
    maze.blocks[:, :] = river_id
    water_mask = np.ones((rows, cols), dtype=bool)
    flow_dir = np.zeros((rows, cols, 2), dtype='float32')
    flow_dir[..., 1] = 1.0
    return maze, WaterBody(water_mask, flow_dir, river_id)


# -- the source graph ---------------------------------------------------------

def test_source_matrix_rows_are_stochastic():
    maze, body = water_room()
    w = WaterAnimation(maze, body, seed=0, start=False)
    row_sums = np.asarray(w._source_matrix.sum(axis=1)).ravel()
    assert np.allclose(row_sums, 1.0)


def test_straight_river_each_cell_is_fed_by_its_upstream_neighbor():
    maze, body = water_room(rows=3, cols=1)
    w = WaterAnimation(maze, body, seed=0, start=False)
    dense = w._source_matrix.toarray()
    # row 0 is the source end: no upstream water neighbour, so it feeds itself
    assert dense[0, 0] == pytest.approx(1.0)
    # row 1 is fed entirely by row 0, directly upstream
    assert dense[1, 0] == pytest.approx(1.0)
    # row 2 is fed entirely by row 1
    assert dense[2, 1] == pytest.approx(1.0)


def test_neighbors_limited_to_water_blocks():
    bt = BlockTypes()
    maze = Maze.filled((3, 1), bt, 'grass')
    Level('room', maze)
    river_id = bt.id_of('river')
    maze.blocks[0, 0] = river_id
    maze.blocks[2, 0] = river_id
    # the middle cell is dry -- it must never supply noise across the gap
    water_mask = np.array([[True], [False], [True]])
    flow_dir = np.zeros((3, 1, 2), dtype='float32')
    flow_dir[..., 1] = 1.0
    body = WaterBody(water_mask, flow_dir, river_id)

    w = WaterAnimation(maze, body, seed=0, start=False)
    dense = w._source_matrix.toarray()
    assert dense.shape == (2, 2)
    # the bottom water cell has no water neighbour across the dry gap, so it
    # falls back to feeding itself rather than being fed by the dry cell
    assert dense[1, 1] == pytest.approx(1.0)


# -- noise update ---------------------------------------------------------------

def test_noise_stays_in_0_1_after_many_steps():
    maze, body = water_room()
    w = WaterAnimation(maze, body, seed=1, start=False)
    for _ in range(50):
        w.advance()
    assert w.noise.min() >= 0.0
    assert w.noise.max() <= 1.0


def test_same_seed_reproduces_the_noise_trajectory():
    maze_a, body_a = water_room()
    maze_b, body_b = water_room()
    a = WaterAnimation(maze_a, body_a, seed=7, start=False)
    b = WaterAnimation(maze_b, body_b, seed=7, start=False)
    for _ in range(5):
        a.advance()
        b.advance()
    assert np.array_equal(a.noise, b.noise)

    c_maze, c_body = water_room()
    c = WaterAnimation(c_maze, c_body, seed=8, start=False)
    for _ in range(5):
        c.advance()
    assert not np.array_equal(a.noise, c.noise)


# -- colour mapping and the display modifier -------------------------------------

def test_color_maps_from_deep_to_foam():
    maze, body = water_room(rows=2, cols=1)
    w = WaterAnimation(maze, body, seed=0, start=False)

    w.noise[:] = 0.0
    w._restyle()
    still_color = w._base_bg[:, :3] + w._color_modifier.bgcolor[:, :3]
    assert np.allclose(still_color, WATER_COLOR_RAMP_RGB[0], atol=1e-5)

    w.noise[:] = 1.0
    w._restyle()
    foam_color = w._base_bg[:, :3] + w._color_modifier.bgcolor[:, :3]
    assert np.allclose(foam_color, WATER_COLOR_RAMP_RGB[-1], atol=1e-5)


def test_noise_settles_toward_the_blue_end_of_the_ramp():
    """The steady-state mean of noise converges to E[random_term] (SELF_WEIGHT
    + NEIGHBOR_WEIGHT damp everything else out -- see WaterAnimation's
    RANDOM_SKEW docstring), so it should settle skewed low, not at 0.5."""
    maze, body = water_room(rows=20, cols=5)
    w = WaterAnimation(maze, body, seed=3, start=False)
    for _ in range(300):
        w.advance()
    assert w.noise.mean() < 0.4


def test_destroy_detaches_from_the_maze_and_is_idempotent():
    maze, body = water_room()
    w = WaterAnimation(maze, body, seed=0, start=False)
    assert w._color_modifier in maze._area_modifiers
    w.destroy()
    assert w._color_modifier not in maze._area_modifiers
    w.destroy()  # idempotent
    assert w._done


def test_area_modifier_reattaches_after_add_scenery_rebuild():
    maze, body = water_room(rows=2, cols=2)
    glyphs = GlyphRegistry()
    layer = SpriteLayer('scenery')
    slot1 = maze.add_scenery(glyphs, layer)

    w = WaterAnimation(maze, body, seed=0, start=False)
    assert w._color_modifier in slot1._modifiers

    # leaving and re-entering the level frees the old slot and builds a new one
    layer.remove_sprites(slot1)
    slot2 = maze.add_scenery(glyphs, layer)
    assert w._color_modifier in slot2._modifiers
    w.advance()  # must recompose against the new slot without raising


# -- integration with the home level's river -------------------------------------

def test_flow_direction_is_unit_length_on_water_cells():
    bt = BlockTypes()
    maze = Maze.filled((100, 300), bt, 'wall', obj_name='home')
    maze.blocks[1:-1, 1:-1] = bt.id_of('grass')
    river = level_001_home._paint_town(maze, bt, seed=0, start=False)

    mags = np.linalg.norm(river.flow_dir[river.mask], axis=-1)
    assert np.allclose(mags, 1.0)


def test_bridge_cells_are_simulated_but_never_painted():
    bt = BlockTypes()
    maze = Maze.filled((100, 300), bt, 'wall', obj_name='home')
    maze.blocks[1:-1, 1:-1] = bt.id_of('grass')
    river = level_001_home._paint_town(maze, bt, seed=0, start=False)
    Level('home', maze)

    river_id = bt.id_of('river')
    bridge_id = bt.id_of('bridge')
    w = river.animation

    bridge_cells = np.flatnonzero((maze.blocks == bridge_id).ravel())
    under_bridge = np.intersect1d(bridge_cells, w._cells)
    assert len(under_bridge) > 0  # the bridge does cross the river

    painted_cells = w._cells[w._paint_index]
    assert not np.isin(under_bridge, painted_cells).any()
    assert np.array_equal(np.sort(painted_cells),
                           np.sort(np.flatnonzero((maze.blocks == river_id).ravel())))

    w.advance()  # simulating the full domain, incl. under the bridge, must not raise
