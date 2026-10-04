import numpy as np

from carriage_return.blocktypes import BlockTypes
from carriage_return.levels import level_001_home
from carriage_return.maze import Maze
from carriage_return.terrain import (
    RIDGE_EAST_WEST, STRAW_HUES, Building, plain_roof, straw_color, thatched_roof)
from carriage_return.terrain.buildings import place_building
from carriage_return.tone_mapping import LUMINANCE_WEIGHTS
from carriage_return.units import CELL_DISPLAY_ASPECT


def test_place_building_returns_its_footprint():
    bt = BlockTypes()
    blocks = np.full((30, 30), bt.id_of('wall'), dtype='uint8')
    blocks[1:-1, 1:-1] = bt.id_of('grass')

    b = place_building(blocks, bt, np.random.RandomState(0), 15, 15, (1, 28), (1, 28))
    assert isinstance(b, Building)
    # every non-floor cell the building stamped lies inside its footprint
    stamped = np.argwhere(blocks != bt.id_of('grass'))[:, ::-1]  # (x, y)
    inner = [p for p in stamped if 0 < p[0] < 29 and 0 < p[1] < 29]
    assert inner and all(b.contains(p) for p in inner)
    assert not b.contains((b.x0 - 1, b.y0)) and not b.contains((b.x0 + b.w, b.y0))


def test_plain_roof_covers_footprint_in_one_colour():
    b = Building(10, 20, 7, 5)
    roof = plain_roof(b, (0.5, 0.2, 0.1), texels_per_cell=4)
    assert roof.albedo.shape == (5 * 4, 7 * 4, 4)
    assert (roof.albedo[..., :3] == np.float32([0.5, 0.2, 0.1])).all()
    assert (roof.albedo[..., 3] == 1).all()


def test_roof_opens_while_the_player_is_inside():
    ss = 4
    b = Building(10, 20, 7, 5)
    roof = plain_roof(b, (0.5, 0.2, 0.1))
    los = np.zeros((40 * ss, 40 * ss), dtype='float32')
    memory = np.zeros_like(los)

    roof.update_sight((12, 22), los, memory, ss)
    assert roof.open
    roof.update_sight((b.x0 + b.w, 22), los, memory, ss)  # just east of it
    assert not roof.open
    roof.update_sight(None, 0.0, memory, ss)               # player elsewhere
    assert not roof.open and roof.seen == 0


def test_roof_seen_and_remembered_come_from_its_footprint():
    ss = 4
    b = Building(10, 20, 7, 5)
    roof = plain_roof(b, (0.5, 0.2, 0.1))
    los = np.zeros((40 * ss, 40 * ss), dtype='float32')
    memory = np.zeros_like(los)

    # sight and memory outside the footprint do not count
    los[0, 0] = memory[0, 0] = 1.0
    roof.update_sight((0, 0), los, memory, ss)
    assert roof.seen == 0 and roof.remembered == 0

    # one wall-face texel of the footprint in view and remembered
    los[b.y0 * ss, b.x0 * ss] = 0.75
    memory[b.y0 * ss, b.x0 * ss] = 0.1
    roof.update_sight((0, 0), los, memory, ss)
    assert roof.seen == 0.75
    assert np.isclose(roof.remembered, 0.1)


def test_home_roofs_every_building():
    bt = BlockTypes()
    maze = Maze.filled((100, 300), bt, 'wall', obj_name='home')
    maze.blocks[1:-1, 1:-1] = bt.id_of('grass')
    _, _, buildings = level_001_home.paint_town(maze, bt, seed=0, start=False)
    assert buildings
    for b in buildings:
        footprint = maze.blocks[b.y0:b.y0 + b.h, b.x0:b.x0 + b.w]
        assert (footprint != bt.id_of('grass')).all()


def test_thatched_roof_has_a_light_west_slope_and_darker_east_slope():
    b = Building(10, 20, 9, 7)
    roof = thatched_roof(b, np.random.RandomState(0))
    res = roof.texels_per_cell
    assert roof.albedo.shape == (7 * res, 9 * res, 4)
    assert (roof.albedo[..., 3] == 1).all()

    lum = roof.albedo[..., :3] @ LUMINANCE_WEIGHTS
    ridge = roof.albedo.shape[1] // 2
    west, east = lum[:, :ridge], lum[:, ridge:]
    assert west.mean() > 1.2 * east.mean()
    # both slopes are straw-coloured: red over green over blue
    for side in (roof.albedo[:, :ridge, :3], roof.albedo[:, ridge:, :3]):
        r, g, bl = side.reshape(-1, 3).mean(axis=0)
        assert r > g > bl


def test_thatch_strands_run_down_the_slopes():
    """Strands run east-west, across the north-south ridge: brightness varies
    far more from row to row than along a row."""
    roof = thatched_roof(Building(0, 0, 9, 7), np.random.RandomState(1))
    lum = roof.albedo[..., :3] @ LUMINANCE_WEIGHTS
    west = lum[1:-1, 1:lum.shape[1] // 2 - 2]
    along = np.abs(np.diff(west, axis=1)).mean()
    across = np.abs(np.diff(west, axis=0)).mean()
    assert across > 2 * along
    assert west.std() > 0.03 * west.mean()   # strands vary in brightness


def test_thatched_roof_is_seeded():
    b = Building(0, 0, 8, 6)
    a1 = thatched_roof(b, np.random.RandomState(3)).albedo
    a2 = thatched_roof(b, np.random.RandomState(3)).albedo
    a3 = thatched_roof(b, np.random.RandomState(4)).albedo
    assert (a1 == a2).all() and not (a1 == a3).all()


def test_east_west_ridge_has_a_light_south_slope_and_strands_running_north_south():
    b = Building(0, 0, 9, 7)
    roof = thatched_roof(b, np.random.RandomState(2), ridge=RIDGE_EAST_WEST)
    res = roof.texels_per_cell
    assert roof.albedo.shape == (7 * res, 9 * res, 4)

    lum = roof.albedo[..., :3] @ LUMINANCE_WEIGHTS
    ridge = lum.shape[0] // 2
    south, north = lum[:ridge], lum[ridge:]   # row 0 is the footprint's south edge
    assert south.mean() > 1.2 * north.mean()

    inner = south[1:-2, 1:-1]
    along = np.abs(np.diff(inner, axis=0)).mean()   # down the slope
    across = np.abs(np.diff(inner, axis=1)).mean()  # from strand to strand
    assert across > 2 * along


def test_unknown_ridge_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        thatched_roof(Building(0, 0, 8, 6), np.random.RandomState(0), ridge='diagonal')


def test_straw_hues():
    rgb = (1.15, 0.80, 0.26)
    assert np.allclose(straw_color(rgb, 'tan'), rgb)

    def saturation(c):
        return c.max() - c.min()
    assert saturation(straw_color(rgb, 'weathered')) < 0.6 * saturation(straw_color(rgb, 'tan'))
    assert (straw_color(rgb, 'bleached') @ LUMINANCE_WEIGHTS
            > straw_color(rgb, 'tan') @ LUMINANCE_WEIGHTS)

    # every hue still thatches with straw: red over green over blue, both slopes
    b = Building(0, 0, 8, 6)
    means = {}
    for hue in STRAW_HUES:
        a = thatched_roof(b, np.random.RandomState(0), hue=hue).albedo[..., :3]
        r, g, bl = a.reshape(-1, 3).mean(axis=0)
        assert r > g > bl
        means[hue] = a.mean(axis=(0, 1))
    # and no two hues look alike
    hues = list(means)
    for i, h1 in enumerate(hues):
        for h2 in hues[i + 1:]:
            assert np.abs(means[h1] - means[h2]).max() > 0.03, (h1, h2)


def test_home_ridges_run_along_the_longer_side_as_shown():
    """Cells are drawn CELL_DISPLAY_ASPECT (0.6) as wide as tall, so the side
    that looks longer is what counts, not the cell count."""
    assert CELL_DISPLAY_ASPECT == 0.6  # the sizes below are chosen for it
    rng = np.random.RandomState(0)
    wide = level_001_home.thatch_roofs([Building(0, 0, 12, 6)] * 10, rng)   # 7.2 x 6 shown
    tall = level_001_home.thatch_roofs([Building(0, 0, 11, 7)] * 10, rng)   # 6.6 x 7 shown
    assert all(_ridge_is_east_west(r) for r in wide)
    assert not any(_ridge_is_east_west(r) for r in tall)

    # a building that looks square has no longer side: either way will do
    square = level_001_home.thatch_roofs([Building(0, 0, 10, 6)] * 40, rng)
    east_west = sum(_ridge_is_east_west(r) for r in square)
    assert 0 < east_west < len(square)


def _ridge_is_east_west(roof):
    """True if the light/dark slopes split the roof south/north rather than
    west/east."""
    lum = roof.albedo[..., :3] @ LUMINANCE_WEIGHTS
    rows, cols = lum.shape
    by_rows = abs(lum[:rows // 2].mean() - lum[rows // 2:].mean())
    by_cols = abs(lum[:, :cols // 2].mean() - lum[:, cols // 2:].mean())
    return by_rows > by_cols
