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
    """Numpy-only visibility provider: everything fully visible."""
    def __init__(self, scene):
        self.shape = scene.field_shape[:2] + (4,)

    def render(self, pos, read=True):
        return np.full(self.shape, 255, dtype='ubyte')


@pytest.fixture
def scene():
    os.chdir(PROJECT_ROOT)  # level1.png is loaded from cwd
    scene = Scene()
    scene.visibility = FakeVisibility(scene)
    player = Player(scene)
    player.location.update(scene.maze, [7, 7])
    Torch(location=(scene.maze, (17, 8)), scene=scene)
    return scene


def test_update_sight(scene):
    v0 = scene.sight.version
    scene.update_sight(1/60.)
    assert scene.sight.version == v0 + 1
    assert scene.sight.data.shape == scene.field_shape
    assert scene.sight.data.max() > 0


def test_memory_decay_is_time_based(scene):
    scene.update_sight(1/60.)
    mem = scene.memory.copy()

    # one 2-second step decays the same as two 1-second steps
    scene_mem_a = mem * scene.MEMORY_DECAY_RATE ** 2.0
    scene_mem_b = (mem * scene.MEMORY_DECAY_RATE ** 1.0) * scene.MEMORY_DECAY_RATE ** 1.0
    assert np.allclose(scene_mem_a, scene_mem_b)

    scene.update_sight(2.0)
    assert np.allclose(scene.memory[:, :, :2], scene_mem_a[:, :, :2])


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


def test_game_model_is_headless():
    """The game model must run without importing any rendering library."""
    result = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, 'agent_helpers', 'check_headless.py')],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'OK' in result.stdout
