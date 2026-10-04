"""Monster spawning: where letters appear, and how often."""
import os
import random

import numpy as np
import pytest

from carriage_return.dm import DungeonMaster
from carriage_return.item import Scroll
from carriage_return.monster import LETTERS, Letter, MonsterSpawner, Wander
from carriage_return.monster.spawn import (MIN_SPAWN_DISTANCE, SPAWN_INTERVAL,
                                           move_distances)
from carriage_return.player import Player
from carriage_return.scene import Scene
from carriage_return.world import Level

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _maze(drawing):
    """A maze from rows of '#' (wall) and '.' (path)."""
    from carriage_return.blocktypes import BlockTypes
    from carriage_return.maze import Maze
    bt = BlockTypes()
    rows = [r.strip() for r in drawing.strip().splitlines()]
    ids = {'#': bt.id_of('wall'), '.': bt.id_of('path')}
    return Maze(np.array([[ids[c] for c in r] for r in rows], dtype='int'), bt)


def _open_maze(shape=(20, 30)):
    rows, cols = shape
    return _maze('\n'.join('#' * cols if r in (0, rows - 1) else '#' + '.' * (cols - 2) + '#'
                           for r in range(rows)))


def _setup(maze, player_pos, seed=0):
    os.chdir(PROJECT_ROOT)  # Scene() loads level1.png from cwd
    scene = Scene()
    scene.set_level(maze)
    player = Player(scene)
    player.location.update(maze, player_pos)
    dm = DungeonMaster(scene)
    spawner = MonsterSpawner(scene, rng=random.Random(seed))
    spawner.attach(dm)
    return scene, player, dm, spawner


def _walkable(maze):
    return maze.blocktypes['walkable'][maze.blocks].astype(bool)


# -- distance ------------------------------------------------------------------

def test_move_distances_count_horizontal_and_vertical_moves_around_walls():
    maze = _maze("""
        #######
        #.#...#
        #.#.#.#
        #...#.#
        #######
    """)
    d = move_distances(_walkable(maze), (1, 1))
    assert d[1, 1] == 0
    assert d[3, 1] == 2
    assert d[3, 3] == 4            # no diagonal shortcut
    assert d[1, 3] == 6            # just across the wall, but a walk around it
    assert d[3, 5] == 10
    assert d[0, 0] == -1           # walls are never reached


def test_move_distances_mark_unreachable_floor():
    maze = _maze("""
        #####
        #.#.#
        #####
    """)
    d = move_distances(_walkable(maze), (1, 1))
    assert d[1, 3] == -1


# -- where ---------------------------------------------------------------------

def test_spawns_are_far_enough_reachable_and_on_open_floor():
    maze = _maze("""
        ##############################
        #............#...............#
        #............#...............#
        #............#...............#
        #............#...............#
        #............#...............#
        #............................#
        ##############################
    """)
    scene, player, dm, spawner = _setup(maze, (11, 1))
    walkable = _walkable(maze)
    dist = move_distances(walkable, (11, 1))
    for _ in range(200):
        m = spawner.spawn()
        x, y = m.location.slot
        assert walkable[y, x]
        assert dist[y, x] >= MIN_SPAWN_DISTANCE
        m.destroy()
    # the cell just across the wall is 2 cells away but a long walk around,
    # so it is a legal spawn even though it is close as the crow flies
    assert dist[1, 14] >= MIN_SPAWN_DISTANCE
    assert [14, 1] in spawner.spawn_cells().tolist()


def test_no_spawn_on_an_occupied_cell():
    # a corridor whose only cell 10 moves from the player is the far end
    maze = _maze("""
        #############
        #...........#
        #############
    """)
    scene, player, dm, spawner = _setup(maze, (1, 1))
    assert spawner.spawn_cells().tolist() == [[11, 1]]

    scroll = Scroll(location=(maze, (11, 1)), scene=scene)
    assert spawner.spawn_cells().tolist() == []
    assert spawner.spawn() is None

    scroll.destroy()
    m = spawner.spawn()
    assert tuple(m.location.slot) == (11, 1)
    assert spawner.spawn_cells().tolist() == []   # the monster occupies it now


def test_spawns_on_the_players_level():
    scene, player, dm, spawner = _setup(_open_maze(), (2, 2))
    other = _open_maze()
    Level('other', other)
    scene.set_level(other)
    player.location.update(other, (3, 3))
    m = spawner.spawn()
    assert m.location.container is other


def test_spawned_monster_is_a_wandering_letter():
    scene, player, dm, spawner = _setup(_open_maze(), (2, 2))
    m = spawner.spawn()
    assert isinstance(m, Letter)
    assert m.type.isa('mob.monster.letter')
    assert m.char in LETTERS and len(m.char) == 1
    assert isinstance(m.behaviour, Wander)
    assert m in scene.monsters


def test_letters_skip_glyphs_other_entities_use():
    assert 't' not in LETTERS     # torch
    assert 'O' not in LETTERS     # hole
    assert len(LETTERS) == 50


# -- when ----------------------------------------------------------------------

def _spawn_turns(dm, scene, turns):
    """Turn numbers (1-based) on which the monster count went up."""
    out = []
    for turn in range(1, turns + 1):
        n = len(scene.monsters)
        dm.end_turn()
        if len(scene.monsters) > n:
            out.append(turn)
    return out


@pytest.mark.parametrize('seed', range(5))
def test_a_monster_spawns_every_spawn_interval(seed):
    scene, player, dm, spawner = _setup(_open_maze((40, 60)), (2, 2), seed=seed)
    turns = _spawn_turns(dm, scene, 400)
    lo, hi = SPAWN_INTERVAL
    gaps = np.diff([0] + turns)
    assert len(turns) >= 400 // hi
    assert all(lo <= g <= hi for g in gaps), gaps


def test_a_player_step_counts_as_a_turn():
    scene, player, dm, spawner = _setup(_open_maze(), (2, 2))
    spawner.countdown = 1
    dm.request_player_move(player, np.array([3, 2]))
    assert len(scene.monsters) == 1


def test_bumping_a_wall_does_not_count():
    scene, player, dm, spawner = _setup(_open_maze(), (1, 2))
    spawner.countdown = 1
    dm.request_player_move(player, np.array([0, 2]))
    assert scene.monsters == []


def test_a_due_spawn_waits_for_a_cell_to_qualify():
    """With nowhere far enough away, the spawn is retried every turn."""
    maze = _maze("""
        #############
        #...........#
        #############
    """)
    scene, player, dm, spawner = _setup(maze, (6, 1))   # nothing 10 moves away
    spawner.countdown = 1
    for _ in range(5):
        dm.end_turn()
    assert scene.monsters == []

    player.location.update(maze, (1, 1))
    dm.end_turn()
    assert len(scene.monsters) == 1
    assert SPAWN_INTERVAL[0] <= spawner.countdown <= SPAWN_INTERVAL[1]
