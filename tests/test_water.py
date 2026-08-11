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
from carriage_return.terrain.meander import natural_extent
from carriage_return.terrain.water import (
    GREENERY_COLOR, GREENERY_MAX_EXTENT, GREENERY_WASH_AMOUNT, MIN_RIVER_WIDTH,
    RIVER_BED_ALBEDO, RIVER_CENTERLINE_DEPTH_M, RIVER_DEEP_WATER_COLOR,
    RIVER_SURFACE_REFLECTANCE, WATER_COLOR_RAMP_RGB,
    RiverBanks, RiverGreenery, WaterAnimation, WaterBody, create_river, paint_river_banks,
    paint_river_greenery, water_bed_color)
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
    assert np.allclose(w._color_modifier.bgcolor[:, :3], WATER_COLOR_RAMP_RGB[0], atol=1e-5)

    w.noise[:] = 1.0
    w._restyle()
    assert np.allclose(w._color_modifier.bgcolor[:, :3], WATER_COLOR_RAMP_RGB[-1], atol=1e-5)


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
    river, _ = level_001_home.paint_town(maze, bt, seed=0, start=False)

    mags = np.linalg.norm(river.flow_dir[river.mask], axis=-1)
    assert np.allclose(mags, 1.0)


def test_flow_direction_follows_start_to_end_regardless_of_coordinate_order():
    bt = BlockTypes()
    river_id = bt.id_of('river')

    # a straight (zero-amplitude) river along y, first start < end...
    maze = Maze.filled((20, 10), bt, 'grass')
    Level('room-down', maze)
    down = create_river(maze, river_id, np.random.RandomState(0), (5, 1), (5, 18),
                         amplitude=0, animate=False)
    assert np.allclose(down.flow_dir[down.mask], [0.0, 1.0])

    # ...then start > end: flow must reverse to match, not stay +y.
    maze = Maze.filled((20, 10), bt, 'grass')
    Level('room-up', maze)
    up = create_river(maze, river_id, np.random.RandomState(0), (5, 18), (5, 1),
                       amplitude=0, animate=False)
    assert np.allclose(up.flow_dir[up.mask], [0.0, -1.0])


def test_bridge_cells_are_simulated_but_never_painted():
    bt = BlockTypes()
    maze = Maze.filled((100, 300), bt, 'wall', obj_name='home')
    maze.blocks[1:-1, 1:-1] = bt.id_of('grass')
    river, _ = level_001_home.paint_town(maze, bt, seed=0, start=False)
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


def test_bridge_cells_show_plain_bridge_color_not_river_depth_wash():
    """create_river washes a depth-dependent colour into maze.bg_color across
    its full mask (see water_bed_color) before paint_town's bridge is ever
    stamped down -- a static mask captured that early would otherwise keep
    tinting the bridge's own cells with river colour forever, since washes
    apply regardless of what a cell's blocktype has since become. bg_color()
    must filter the wash live against blocktype_id instead, so a bridge cell
    reads as plain bridge colour (bg_color_var noise aside)."""
    bt = BlockTypes()
    maze = Maze.filled((100, 300), bt, 'wall', obj_name='home')
    maze.blocks[1:-1, 1:-1] = bt.id_of('grass')
    level_001_home.paint_town(maze, bt, seed=0, start=False)

    bridge_id = bt.id_of('bridge')
    bridge_mask = maze.blocks == bridge_id
    assert bridge_mask.any()  # the bridge does cross the river

    bridge_base = bt['bg_color'][bridge_id][:3]
    bridge_colors = maze.bg_color[bridge_mask][:, :3]
    # bg_color_var=0.02 for bridge cells is the only expected deviation from
    # the flat blocktype colour; the depth wash (or a bridge painted over
    # blue-ish deep water) would shift this far more than per-cell noise ever
    # would.
    assert np.allclose(bridge_colors, bridge_base, atol=0.15)


# -- variable width and the depth field -------------------------------------------

def straight_river(width, rows=80, cols=20, seed=2, wavelength=130):
    """A river with a fixed (amplitude=0) centreline, so any width change
    seen along its length comes only from the width-meander itself."""
    bt = BlockTypes()
    maze = Maze.filled((rows, cols), bt, 'grass')
    Level('room', maze)
    river_id = bt.id_of('river')
    river = create_river(maze, river_id, np.random.RandomState(seed), (cols // 2, 2),
                          (cols // 2, rows - 3), amplitude=0, width=width,
                          wavelength=wavelength, animate=False)
    return maze, river


def test_river_width_varies_around_the_nominal_width():
    nominal = 12
    _, river = straight_river(nominal)
    # it actually wanders, rather than sitting flat at the nominal width...
    assert river.width.std() > 0
    assert (river.width != nominal).any()
    # ...within something like the 10-20% band the plan calls for (loose
    # bounds -- the underlying OU process only *typically* stays this
    # close, so this checks it's in the right ballpark, not exact).
    assert river.width.min() >= nominal * 0.6
    assert river.width.max() <= nominal * 1.4


def test_river_width_never_drops_below_the_clamped_minimum():
    # a tiny nominal width and short width-wavelength, so the meander
    # actually dips to (and is clamped at) MIN_RIVER_WIDTH somewhere along a
    # reasonably long river -- not just trivially never approaching it.
    _, river = straight_river(width=3, wavelength=8)
    assert river.width.min() == MIN_RIVER_WIDTH


def test_width_and_half_width_are_indexed_like_the_path_centerline():
    _, river = straight_river(12)
    assert len(river.width) == len(river.path.centerline)
    assert len(river.half_width) == len(river.path.centerline)
    assert np.array_equal(river.half_width, river.width // 2)


def test_depth_is_one_at_centerline_and_small_at_the_band_edges():
    _, river = straight_river(12)
    path = river.path
    for i in (5, len(path.centerline) // 2, len(path.centerline) - 6):
        coord = path.lo + i
        c = int(path.centerline[i])
        half = int(river.half_width[i])
        w = int(river.width[i])
        row_or_col = river.depth[coord] if path.axis == 'y' else river.depth[:, coord]

        assert row_or_col[c] == pytest.approx(1.0)
        assert row_or_col[c - half] < 0.3
        assert row_or_col[c - half + w - 1] < 0.3


def test_depth_is_zero_outside_the_mask():
    _, river = straight_river(12)
    assert np.all(river.depth[~river.mask] == 0.0)


def test_mask_band_stays_contiguous_per_row_with_variable_width():
    """paint_town's bridge sizing derives the river's extent at a row band
    from river.mask directly, via min/max over the nonzero columns -- which
    only finds the right span if each row's water forms one unbroken run.
    Variable width (rasterized per-row, independently) must not introduce
    gaps within a row."""
    _, river = straight_river(12)
    for coord in range(river.path.lo, river.path.hi + 1):
        row = river.mask[coord] if river.path.axis == 'y' else river.mask[:, coord]
        cols = np.flatnonzero(row)
        if len(cols):
            assert cols.max() - cols.min() + 1 == len(cols)

    # and paint_town itself (the real caller of this logic, with its own
    # meandering centreline) still runs end-to-end and paints a bridge over
    # a nonempty stretch of the (now variable-width) river.
    bt = BlockTypes()
    maze = Maze.filled((100, 300), bt, 'wall', obj_name='home')
    maze.blocks[1:-1, 1:-1] = bt.id_of('grass')
    river, _ = level_001_home.paint_town(maze, bt, seed=0, start=False)
    bridge_id = bt.id_of('bridge')
    assert (maze.blocks == bridge_id).any()


# -- depth-dependent base colour -------------------------------------------------

def test_water_bed_color_endpoints_match_albedo_and_deep_water():
    # surface_reflectance=0 isolates the Beer-Lambert mix itself; its own
    # dimming effect is covered separately below.
    color = water_bed_color(np.array([0.0, RIVER_CENTERLINE_DEPTH_M], dtype='float32'),
                             surface_reflectance=0.0)
    assert np.allclose(color[0], RIVER_BED_ALBEDO, atol=1e-5)
    # centreline depth doesn't fully reach deep water (finite absorption),
    # but should already sit much closer to it than to the bed albedo.
    dist_to_deep = np.linalg.norm(color[1] - RIVER_DEEP_WATER_COLOR)
    dist_to_bed = np.linalg.norm(color[1] - RIVER_BED_ALBEDO)
    assert dist_to_deep < dist_to_bed


def test_water_bed_color_surface_reflectance_dims_the_whole_mix():
    """A fraction of light never makes it past the surface at all, so the
    reflectance-aware result should be a uniform darkening of the plain
    Beer-Lambert blend, regardless of depth."""
    depth = np.array([0.0, 0.3, RIVER_CENTERLINE_DEPTH_M], dtype='float32')
    plain = water_bed_color(depth, surface_reflectance=0.0)
    dimmed = water_bed_color(depth, surface_reflectance=RIVER_SURFACE_REFLECTANCE)
    assert np.allclose(dimmed, plain * (1.0 - RIVER_SURFACE_REFLECTANCE))


def test_water_bed_color_is_never_nan():
    depth = np.array([0.0, 0.3, RIVER_CENTERLINE_DEPTH_M, -0.0], dtype='float32')
    assert np.isfinite(water_bed_color(depth)).all()


def test_shallow_river_cells_read_warmer_than_deep_cells():
    """Shallow (bank-adjacent) cells should show more of the bed's warm,
    red-heavy albedo; deep (centreline) cells should read cooler and bluer --
    the whole point of the depth-dependent colour model."""
    maze, river = straight_river(12)
    path = river.path
    i = len(path.centerline) // 2
    coord = path.lo + i
    c = int(path.centerline[i])
    half = int(river.half_width[i])

    bg = maze.bg_color
    shallow = bg[coord, c - half] if path.axis == 'y' else bg[c - half, coord]
    deep = bg[coord, c] if path.axis == 'y' else bg[c, coord]

    # red dominates the shallow, bed-toned cell; blue dominates the deep one.
    assert shallow[0] > shallow[2]
    assert deep[2] > deep[0]
    # and the deep cell reads distinctly bluer than the shallow one.
    assert deep[2] - deep[0] > shallow[2] - shallow[0]


def test_river_wash_is_applied_before_animation_attaches():
    """create_river washes depth colour onto the maze before body.animate()
    ever runs -- whether animate=True (immediately) or animate=False with the
    caller attaching later (paint_town's bridge-over-river ordering). The
    wash is registered on the maze itself (see Maze.wash_bg_color), so
    maze.bg_color must reflect the depth gradient, not the flat 'river'
    blocktype colour, regardless of when animation attaches."""
    bt = BlockTypes()
    maze = Maze.filled((80, 20), bt, 'grass')
    Level('room', maze)
    river_id = bt.id_of('river')
    flat_river_color = np.array(bt.get('river')['bg_color'][:3])

    river = create_river(maze, river_id, np.random.RandomState(2), (10, 2), (10, 77),
                          amplitude=0, width=12, animate=True, start_animation=False)

    base_bg = maze.bg_color[river.mask][:, :3]
    # every painted cell's colour differs from the flat blocktype colour --
    # the depth wash, not the fallback, is what animation is layered on.
    assert not np.allclose(base_bg, flat_river_color, atol=1e-4)
    # and it varies across cells (shallow vs deep), not just uniformly offset.
    assert base_bg.std(axis=0).sum() > 0


def test_river_wash_survives_animate_false_then_animate_later():
    """Mirrors paint_town's real ordering: create_river(animate=False), some
    other paint happens, then river.animate(...) is called afterward. The
    depth wash is applied inside create_river as a persistent maze-level
    wash, so it's still there in maze.bg_color independent of when (or
    whether) .animate() is subsequently called."""
    bt = BlockTypes()
    maze = Maze.filled((80, 20), bt, 'grass')
    Level('room', maze)
    river_id = bt.id_of('river')
    flat_river_color = np.array(bt.get('river')['bg_color'][:3])

    river = create_river(maze, river_id, np.random.RandomState(2), (10, 2), (10, 77),
                          amplitude=0, width=12, animate=False)
    # nothing else paints over the river here (unlike paint_town's bridge),
    # but the ordering under test is: wash happens inside create_river,
    # independent of when/whether .animate() is subsequently called.
    river.animate(maze, seed=0, start=False)

    base_bg = maze.bg_color[river.mask][:, :3]
    assert not np.allclose(base_bg, flat_river_color, atol=1e-4)


# -- sandy banks --------------------------------------------------------------------

def bank_room(width=4, rows=30, cols=20, seed=2, wavelength=40, cx=None):
    """A straight (amplitude=0) river down the middle of a plain grass room,
    for exercising paint_river_banks without paint_town's full complexity."""
    bt = BlockTypes()
    if cx is None:
        cx = cols // 2
    maze = Maze.filled((rows, cols), bt, 'grass')
    Level('room', maze)
    river = create_river(maze, bt.id_of('river'), np.random.RandomState(seed), (cx, 2),
                          (cx, rows - 3), amplitude=0, width=width, wavelength=wavelength,
                          animate=False)
    return bt, maze, river


def test_river_banks_are_at_most_one_cell_wide():
    bt, maze, river = bank_room()
    banks = paint_river_banks(maze, bt, river, np.random.RandomState(0))
    assert banks.left.min() >= 0 and banks.left.max() <= 1
    assert banks.right.min() >= 0 and banks.right.max() <= 1


def test_river_banks_are_indexed_like_the_path_centerline():
    bt, maze, river = bank_room()
    banks = paint_river_banks(maze, bt, river, np.random.RandomState(0))
    assert len(banks.left) == len(river.path.centerline)
    assert len(banks.right) == len(river.path.centerline)


def test_river_banks_are_adjacent_to_the_water():
    bt, maze, river = bank_room()
    paint_river_banks(maze, bt, river, np.random.RandomState(0))

    sand_mask = maze.blocks == bt.id_of('sand')
    assert sand_mask.any()  # the scenario actually exercises the guard

    padded = np.pad(river.mask, 1)
    neighbor_is_water = (padded[:-2, 1:-1] | padded[2:, 1:-1] |
                          padded[1:-1, :-2] | padded[1:-1, 2:])
    assert neighbor_is_water[sand_mask].all()


def test_river_banks_never_overwrite_non_grass():
    bt, maze, river = bank_room()

    # plant a dirt cell exactly where the left bank at row-index 5 would
    # otherwise land, before banks are painted -- it must survive untouched
    # and count as nothing placed there.
    i = 5
    c = int(river.path.centerline[i])
    half = int(river.half_width[i])
    near_edge = c - half
    blocked_col = near_edge - 1
    row = river.path.lo + i
    maze.blocks[row, blocked_col] = bt.id_of('dirt')

    banks = paint_river_banks(maze, bt, river, np.random.RandomState(0))

    assert maze.blocks[row, blocked_col] == bt.id_of('dirt')
    assert banks.left[i] == 0


def test_river_banks_stay_within_maze_bounds_near_a_room_edge():
    # the river hugs the room's left edge, so its left bank would run
    # negative (and, unclipped, numpy would silently wrap to the far column)
    # if paint_river_banks didn't clip to the maze's own bounds.
    bt, maze, river = bank_room(width=4, cols=8, cx=1)
    paint_river_banks(maze, bt, river, np.random.RandomState(0))
    assert not (maze.blocks[:, -1] == bt.id_of('sand')).any()


def test_river_banks_max_extent_zero_places_nothing():
    bt, maze, river = bank_room()
    banks = paint_river_banks(maze, bt, river, np.random.RandomState(0), max_extent=0)
    assert (banks.left == 0).all()
    assert (banks.right == 0).all()
    assert not (maze.blocks == bt.id_of('sand')).any()


def test_river_banks_placed_counts_match_actual_sand_cells():
    bt, maze, river = bank_room()
    banks = paint_river_banks(maze, bt, river, np.random.RandomState(0))
    assert (maze.blocks == bt.id_of('sand')).sum() == banks.left.sum() + banks.right.sum()


# -- integration with the home level's river -----------------------------------------

def test_homes_town_has_sandy_banks():
    bt = BlockTypes()
    maze = Maze.filled((100, 300), bt, 'wall', obj_name='home')
    maze.blocks[1:-1, 1:-1] = bt.id_of('grass')
    river, _ = level_001_home.paint_town(maze, bt, seed=0, start=False)

    assert (maze.blocks == bt.id_of('sand')).any()
    assert isinstance(river.banks, RiverBanks)
    assert len(river.banks.left) == len(river.path.centerline)
    assert len(river.banks.right) == len(river.path.centerline)


# -- lush greenery beyond the banks --------------------------------------------------

def greenery_room(width=4, rows=60, cols=20, seed=2, wavelength=40, cx=None,
                   bank_seed=0, greenery_seed=1, max_extent=GREENERY_MAX_EXTENT):
    """A straight river with sandy banks already painted, plus the greenery
    painted just past them -- for exercising paint_river_greenery without
    paint_town's full complexity."""
    bt, maze, river = bank_room(width=width, rows=rows, cols=cols, seed=seed,
                                 wavelength=wavelength, cx=cx)
    banks = paint_river_banks(maze, bt, river, np.random.RandomState(bank_seed))
    greenery = paint_river_greenery(maze, bt, river, banks, np.random.RandomState(greenery_seed),
                                     max_extent=max_extent)
    return bt, maze, river, banks, greenery


def test_greenery_stays_within_the_max_extent_bound():
    bt, maze, river, banks, greenery = greenery_room()
    assert greenery.left.min() >= 0 and greenery.left.max() <= GREENERY_MAX_EXTENT
    assert greenery.right.min() >= 0 and greenery.right.max() <= GREENERY_MAX_EXTENT


def test_greenery_is_indexed_like_the_path_centerline():
    bt, maze, river, banks, greenery = greenery_room()
    assert len(greenery.left) == len(river.path.centerline)
    assert len(greenery.right) == len(river.path.centerline)


def test_greenery_only_touches_currently_grass_cells():
    bt, maze, river, banks, greenery = greenery_room()
    # greenery never changes the blocktype -- grass stays grass -- so this is
    # really just confirming its counts only ever describe cells that are
    # (still) grass, the same guard paint_river_banks itself uses.
    grass_id = bt.id_of('grass')
    for i in np.flatnonzero(greenery.left > 0):
        c = int(river.path.centerline[i])
        half = int(river.half_width[i])
        near_edge = c - half
        start = near_edge - int(banks.left[i])
        row = river.path.lo + i
        cells = maze.blocks[row, max(start - int(greenery.left[i]), 0):start]
        assert (cells == grass_id).all()


def test_greenery_placed_counts_start_exactly_past_the_actual_bank_edge():
    """Recomputes what paint_river_greenery *should* place -- independently,
    straight from river.banks and the maze's own grass footprint -- and
    checks its returned counts match exactly. This is the direct check that
    a band starts right where the actually-placed sand ends, not at the
    water's edge itself (which would double-count or skip cells whenever a
    bank got clipped short of BANK_MAX_EXTENT)."""
    bt, maze, river = bank_room(width=4, rows=60, cols=20, seed=2, wavelength=40)
    banks = paint_river_banks(maze, bt, river, np.random.RandomState(0))

    length = len(river.path.centerline)
    check_rng = np.random.RandomState(1)
    left_wanted = natural_extent(check_rng, length, GREENERY_MAX_EXTENT)
    right_wanted = natural_extent(check_rng, length, GREENERY_MAX_EXTENT)

    greenery = paint_river_greenery(maze, bt, river, banks, np.random.RandomState(1))

    grass_id = bt.id_of('grass')
    rows, cols = maze.blocks.shape
    for i in range(length):
        c = int(river.path.centerline[i])
        half = int(river.half_width[i])
        w = int(river.width[i])
        near_edge = c - half
        far_edge = near_edge + w
        row = river.path.lo + i

        left_start = near_edge - int(banks.left[i])
        lo, hi = max(left_start - int(left_wanted[i]), 0), min(left_start, cols)
        expected_left = int((maze.blocks[row, lo:hi] == grass_id).sum()) if hi > lo else 0
        assert greenery.left[i] == expected_left

        right_start = far_edge + int(banks.right[i])
        lo, hi = max(right_start, 0), min(right_start + int(right_wanted[i]), cols)
        expected_right = int((maze.blocks[row, lo:hi] == grass_id).sum()) if hi > lo else 0
        assert greenery.right[i] == expected_right


def test_greenery_wash_is_strong_near_the_bank_and_fades_to_nothing_at_the_outer_edge():
    bt, maze, river, banks, greenery = greenery_room(max_extent=3)
    grass_color = np.asarray(bt.get('grass')['bg_color'][:3])

    # a position where a full band actually reached the maximum extent, so
    # both ends of the taper are exercised.
    idxs = np.flatnonzero(greenery.left == GREENERY_MAX_EXTENT)
    assert len(idxs) > 0, "no full-length band placed on the left -- widen the room/seed"
    i = idxs[0]
    c = int(river.path.centerline[i])
    half = int(river.half_width[i])
    near_edge = c - half
    bank_left = int(banks.left[i])
    row = river.path.lo + i

    near_col = near_edge - bank_left - 1               # right past the bank
    far_col = near_edge - bank_left - GREENERY_MAX_EXTENT  # outer edge of the band

    bg = maze.bg_color
    near_color = np.asarray(bg[row, near_col, :3])
    far_color = np.asarray(bg[row, far_col, :3])

    # near the bank: distinctly greener than plain grass, and closer to the
    # lush colour than a faint tint would be.
    assert near_color[1] - grass_color[1] > 0.05
    # at the outer edge: the field itself already blended back to grass's own
    # colour, so washing it in has (up to noise) no visible effect.
    assert abs(far_color[1] - grass_color[1]) < 0.02
    assert near_color[1] > far_color[1]


def test_greenery_wash_amount_and_color_are_distinct_from_the_parched_grass_wash():
    from carriage_return.terrain.grass import GRASS_WASH_AMOUNT, GRASS_WASH_RAMP_RGB
    assert GREENERY_WASH_AMOUNT > GRASS_WASH_AMOUNT
    # richer/more saturated green than the parched-patch ramp's own deep-green
    # stop, which sits close to plain grass by design.
    assert GREENERY_COLOR[1] > GRASS_WASH_RAMP_RGB[0][1]


# -- integration with the home level's river -----------------------------------------

def test_homes_town_has_lush_greenery_past_its_banks():
    bt = BlockTypes()
    maze = Maze.filled((100, 300), bt, 'wall', obj_name='home')
    maze.blocks[1:-1, 1:-1] = bt.id_of('grass')
    river, _ = level_001_home.paint_town(maze, bt, seed=0, start=False)

    assert isinstance(river.greenery, RiverGreenery)
    assert len(river.greenery.left) == len(river.path.centerline)
    assert len(river.greenery.right) == len(river.path.centerline)
    assert river.greenery.left.max() <= GREENERY_MAX_EXTENT
    assert river.greenery.right.max() <= GREENERY_MAX_EXTENT
    assert river.greenery.left.sum() + river.greenery.right.sum() > 0
