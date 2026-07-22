import subprocess
import sys
import os

import numpy as np
import pytest

from carriage_return.scene import Scene
from carriage_return.player import Player
from carriage_return.item import Scroll, Torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeVisibility:
    """Numpy-only visibility provider: everything fully visible.

    Reads the field shape per call rather than caching it, because
    scene.set_level resizes the field when the level changes.
    """
    def __init__(self, scene):
        self.scene = scene

    def render(self, pos, read=True):
        return np.full(self.scene.field_shape[:2] + (4,), 255, dtype='ubyte')


def _auto_visibility(scene):
    """Inject a FakeVisibility onto every level the scene shows, as the real
    renderer does on level_changed."""
    scene.level_changed.connect(lambda: setattr(scene.level, 'visibility',
                                                 FakeVisibility(scene)))
    scene.level.visibility = FakeVisibility(scene)


@pytest.fixture
def scene():
    os.chdir(PROJECT_ROOT)  # level1.png is loaded from cwd
    scene = Scene()
    _auto_visibility(scene)
    player = Player(scene)
    player.location.update(scene.maze, [7, 7])
    Torch(location=(scene.maze, (17, 8)), scene=scene)
    return scene


def test_update_sight(scene):
    v0 = scene.sight.version
    scene.update_sight(1/60.)
    assert scene.sight.version == v0 + 1
    # sight packs RGBA now: RGB linear HDR light, A a memory overlay
    assert scene.sight.data.shape == scene.field_shape[:2] + (4,)
    # the fixture is lit and fully visible, so the HDR RGB is non-zero
    assert scene.sight.data.max() > 0


def test_memory_decay_is_time_based(scene):
    scene.update_sight(1/60.)

    # update_sight now re-accumulates max(memory, current_lit) BEFORE decaying,
    # so to isolate pure decay we dominate the accumulation: set memory far
    # above any lit value, so the max() keeps it and only the decay acts.
    scene.memory[:] = 1e6
    mem = scene.memory.copy()

    # one 2-second step decays the same as two 1-second steps
    scene_mem_a = mem * scene.MEMORY_DECAY_RATE ** 2.0
    scene_mem_b = (mem * scene.MEMORY_DECAY_RATE ** 1.0) * scene.MEMORY_DECAY_RATE ** 1.0
    assert np.allclose(scene_mem_a, scene_mem_b)

    scene.update_sight(2.0)
    assert np.allclose(scene.memory, scene_mem_a)


def test_write_message(scene):
    n0 = len(scene.log.lines)
    version = scene.log.version
    scene.write("hello")
    assert scene.log.lines[n0:] == ["hello"]
    assert scene.log.version == version + 1


def test_scroll_read_opens_pager(scene):
    from carriage_return.input import InputDispatcher, KeyPress
    scroll = Scroll(location=(scene.maze, (5, 5)), scene=scene)
    assert scroll in scene.items

    InputDispatcher.reset()
    dispatcher = InputDispatcher()
    try:
        session = scroll.read(scene.player)  # opens a pager, does not destroy
        assert dispatcher.handlers[-1] is session
        assert len(scene.grids) == 1
        grid = list(scene.grids)[0]
        text = '\n'.join(''.join(grid.registry.chars[i] for i in grid.glyph[r])
                         for r in range(grid.shape[0]))
        assert scroll.description in text

        session.post(KeyPress('Escape'))
        session.join(10)
        assert session.finished.is_set() and session.error is None

        # reading no longer consumes the scroll (the pages end the joke)
        assert scroll in scene.items
        assert scroll in scene.maze.inventory[(5, 5)]
        assert len(scene.grids) == 0
        assert dispatcher.handlers == []
    finally:
        InputDispatcher.reset()


def _small_maze(shape=(12, 20)):
    """A maze of open path with a wall border, independent of level1.png."""
    from carriage_return.blocktypes import BlockTypes
    from carriage_return.maze import Maze
    bt = BlockTypes()
    blocks = np.full(shape, bt.id_of('path'), dtype='int')
    blocks[0, :] = blocks[-1, :] = bt.id_of('wall')
    blocks[:, 0] = blocks[:, -1] = bt.id_of('wall')
    return Maze(blocks, bt)


def test_set_level_swaps_maze_and_resizes_fields(scene):
    ss = scene.supersample
    maze = _small_maze()

    scene.set_level(maze)

    assert scene.maze is maze
    assert scene.field_shape == (12 * ss, 20 * ss, 3)
    assert scene.memory.shape == scene.field_shape[:2]
    assert scene.line_of_sight.shape == scene.field_shape
    assert scene.sight.data.shape == scene.field_shape[:2] + (4,)

    # the sight pipeline still runs against the new level
    scene.player.location.update(maze, [5, 5])
    scene.update_sight(1 / 60.)
    assert scene.sight.data.shape == scene.field_shape[:2] + (4,)


def test_set_level_frees_the_previous_scenery(scene):
    """Switching levels must not leak the outgoing maze's sprites."""
    layer = scene.sprite_layers['scenery']
    assert len(layer) == int(np.prod(scene.maze.shape))

    small = _small_maze((12, 20))
    scene.set_level(small)
    assert len(layer) == 12 * 20
    assert len(layer.slots) == 1

    # switching back and forth does not accumulate
    scene.set_level(_small_maze((8, 9)))
    assert len(layer) == 8 * 9
    assert len(layer.slots) == 1


def test_set_level_fires_level_changed(scene):
    calls = []
    scene.level_changed.connect(lambda: calls.append(scene.maze))
    maze = _small_maze()
    scene.set_level(maze)
    # fired once, after scene state is already consistent
    assert calls == [maze]


def test_game_model_is_headless():
    """The game model must run without importing any rendering library."""
    result = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, 'agent_helpers', 'check_headless.py')],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'OK' in result.stdout
