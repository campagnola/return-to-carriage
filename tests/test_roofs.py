import numpy as np

from carriage_return.blocktypes import BlockTypes
from carriage_return.levels import level_001_home
from carriage_return.maze import Maze
from carriage_return.terrain import Building, plain_roof
from carriage_return.terrain.buildings import place_building


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
