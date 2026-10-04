"""Monsters: entities in the maze that take a turn after each player step."""
import os
import random

import numpy as np
import pytest

from carriage_return.dm import DungeonMaster
from carriage_return.entity import Entity
from carriage_return.monster import Idle, Monster, Wander
from carriage_return.player import Player
from carriage_return.scene import Scene
from carriage_return.world import Level

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _small_maze(shape=(12, 20)):
    """A maze of open path with a wall border, independent of level1.png."""
    from carriage_return.blocktypes import BlockTypes
    from carriage_return.maze import Maze
    bt = BlockTypes()
    blocks = np.full(shape, bt.id_of('path'), dtype='int')
    blocks[0, :] = blocks[-1, :] = bt.id_of('wall')
    blocks[:, 0] = blocks[:, -1] = bt.id_of('wall')
    return Maze(blocks, bt)


class Step:
    """A behaviour that always asks for the same step."""
    def __init__(self, dx, dy):
        self.step = (dx, dy)

    def choose_step(self, monster, dm):
        return self.step


@pytest.fixture
def world():
    """A scene showing a small walled maze, with the player at (5, 5)."""
    os.chdir(PROJECT_ROOT)  # Scene() loads level1.png from cwd
    scene = Scene()
    maze = _small_maze()
    scene.set_level(maze)
    player = Player(scene)
    player.location.update(maze, (5, 5))
    return scene, maze, player, DungeonMaster(scene)


def _pos(entity):
    return tuple(int(v) for v in entity.location.slot)


# -- being in the maze ---------------------------------------------------------

def test_monster_stands_in_the_maze_inventory(world):
    scene, maze, player, dm = world
    m = Monster(location=(maze, (8, 5)), scene=scene)
    assert m in scene.monsters
    assert m in maze.inventory[(8, 5)]
    assert m.type.isa('mob.monster')
    assert m.level is maze.level
    assert isinstance(m.behaviour, Idle)


def test_monster_blocks_the_player(world):
    scene, maze, player, dm = world
    Monster(location=(maze, (6, 5)), scene=scene)
    assert not dm.walkable((6, 5))
    dm.request_player_move(player, np.array([6, 5]))
    assert _pos(player) == (5, 5)


def test_destroy_removes_the_monster(world):
    scene, maze, player, dm = world
    m = Monster(location=(maze, (8, 5)), scene=scene)
    m.destroy()
    assert m not in scene.monsters
    assert m not in maze.inventory[(8, 5)]
    assert np.isnan(m.sprite.sprite.position).all()


# -- moving --------------------------------------------------------------------

def test_monster_moves_onto_open_floor(world):
    scene, maze, player, dm = world
    m = Monster(location=(maze, (8, 5)), scene=scene)
    assert dm.request_monster_move(m, (9, 6))
    assert _pos(m) == (9, 6)
    assert m in maze.inventory[(9, 6)]
    assert m not in maze.inventory[(8, 5)]


@pytest.mark.parametrize('start, target', [
    ((1, 3), (0, 3)),     # the left-hand wall
    ((1, 3), (-1, 3)),    # off the map (would wrap around in numpy)
    ((18, 3), (20, 3)),   # off the map
    ((6, 5), (5, 5)),     # the player
    ((8, 8), (9, 8)),     # another monster
])
def test_monster_move_is_refused(world, start, target):
    scene, maze, player, dm = world
    Monster(location=(maze, (9, 8)), scene=scene)
    m = Monster(location=(maze, start), scene=scene)
    assert not dm.request_monster_move(m, target)
    assert _pos(m) == start


def test_walkable_is_false_outside_the_maze(world):
    scene, maze, player, dm = world
    for pos in [(-1, 3), (3, -1), (20, 3), (3, 12)]:
        assert not dm.walkable(pos)
    assert dm.walkable((3, 3))


# -- turns ---------------------------------------------------------------------

def test_each_player_step_gives_monsters_one_turn(world):
    scene, maze, player, dm = world
    m = Monster(location=(maze, (10, 3)), scene=scene, behaviour=Step(1, 0))
    dm.request_player_move(player, np.array([6, 5]))
    assert _pos(m) == (11, 3)
    dm.request_player_move(player, np.array([7, 5]))
    assert _pos(m) == (12, 3)


def test_bumping_a_wall_is_not_a_turn(world):
    scene, maze, player, dm = world
    player.location.update(maze, (1, 5))
    m = Monster(location=(maze, (10, 3)), scene=scene, behaviour=Step(1, 0))
    dm.request_player_move(player, np.array([0, 5]))
    assert _pos(player) == (1, 5)
    assert _pos(m) == (10, 3)


def test_bumping_a_wall_does_not_retrigger_what_is_underfoot(world):
    """Walking into a wall used to 'move' the player in place, re-running
    on_walked_on for everything on its cell (a walk-on portal end would ask to
    traverse again)."""
    scene, maze, player, dm = world

    class Tripwire(Entity):
        def __init__(self):
            Entity.__init__(self, entity_type='test.tripwire')
            self.trips = 0

        def on_walked_on(self, mover, dm):
            self.trips += 1

    from carriage_return.location import Location
    wire = Tripwire()
    wire.location = Location(wire, maze, (1, 5))
    dm.request_player_move(player, np.array([1, 5]))
    assert wire.trips == 1
    dm.request_player_move(player, np.array([0, 5]))  # the wall
    assert wire.trips == 1


def test_monsters_off_the_players_level_are_frozen(world):
    scene, maze, player, dm = world
    other = _small_maze()
    Level('other', other)
    m = Monster(location=(other, (3, 3)), scene=scene, behaviour=Step(1, 0))
    dm.end_turn()
    assert _pos(m) == (3, 3)

    # once the player is on its level, it acts again
    scene.set_level(other)
    player.location.update(other, (8, 8))
    dm.end_turn()
    assert _pos(m) == (4, 3)


def test_monsters_act_in_the_order_they_were_added(world):
    """The first monster steps into the cell the second would have wanted."""
    scene, maze, player, dm = world
    a = Monster(location=(maze, (10, 3)), scene=scene, behaviour=Step(1, 0))
    b = Monster(location=(maze, (12, 3)), scene=scene, behaviour=Step(-1, 0))
    dm.end_turn()
    assert _pos(a) == (11, 3)
    assert _pos(b) == (12, 3)


# -- behaviours ----------------------------------------------------------------

def test_wander_is_repeatable_with_a_seed_and_stays_on_open_floor(world):
    scene, maze, player, dm = world

    def walk(seed):
        m = Monster(location=(maze, (14, 8)), scene=scene,
                    behaviour=Wander(random.Random(seed)))
        path = []
        for _ in range(40):
            m.take_turn(dm)
            x, y = _pos(m)
            assert maze.blocktype_at(y, x)['walkable']
            assert (x, y) != _pos(player)
            path.append((x, y))
        m.destroy()
        return path

    path = walk(1)
    assert walk(1) == path
    assert len(set(path)) > 1


def test_wander_stays_put_when_boxed_in(world):
    scene, maze, player, dm = world
    # (1, 1) is a corner: its open neighbours are (2, 1), (1, 2), (2, 2)
    m = Monster(location=(maze, (1, 1)), scene=scene, behaviour=Wander(random.Random(0)))
    for pos in [(2, 1), (1, 2), (2, 2)]:
        Monster(location=(maze, pos), scene=scene)
    assert m.behaviour.choose_step(m, dm) is None
    assert not m.take_turn(dm)
    assert _pos(m) == (1, 1)


# -- display -------------------------------------------------------------------

def test_sprite_hides_while_its_level_is_not_shown(world):
    scene, maze, player, dm = world
    m = Monster(location=(maze, (8, 5)), scene=scene)
    assert tuple(m.sprite.sprite.position[0][:2]) == (8, 5)

    scene.set_level(_small_maze())
    assert np.isnan(m.sprite.sprite.position).all()

    scene.set_level(maze)
    assert tuple(m.sprite.sprite.position[0][:2]) == (8, 5)


def test_dungeon_spawns_a_wandering_monster_on_open_floor():
    os.chdir(PROJECT_ROOT)
    from carriage_return.levels import build_world
    scene = Scene()
    world = build_world(scene)
    dungeon = world.levels['dungeon']
    [m] = scene.monsters
    assert m.level is dungeon
    assert isinstance(m.behaviour, Wander)
    x, y = _pos(m)
    assert dungeon.maze.blocktype_at(y, x)['walkable']
