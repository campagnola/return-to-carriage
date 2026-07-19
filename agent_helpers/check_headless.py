"""Prove the game model runs with no rendering library loaded.

Builds a Scene with a fake numpy visibility provider, runs sight updates,
then exercises the full game-side dialog pipeline (reading a scroll opens a
pager on its own thread, with a grid in scene.grids) and asserts that no
rendering/GUI module was ever imported.

Run: python agent_helpers/check_headless.py
"""
import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)  # level1.png is loaded from cwd

from carriage_return.scene import Scene
from carriage_return.player import Player
from carriage_return.item import Scroll, Torch
from carriage_return.input import InputDispatcher, KeyPress


class FakeVisibility:
    """Numpy-only stand-in for graphics.ShadowRenderer."""
    def __init__(self, scene):
        self.shape = scene.field_shape[:2] + (4,)

    def render(self, pos, read=True):
        return np.full(self.shape, 255, dtype='ubyte')


def main():
    scene = Scene()
    scene.visibility = FakeVisibility(scene)

    player = Player(scene)
    player.location.update(scene.maze, [7, 7])

    torch = Torch(location=(scene.maze, (17, 8)), scene=scene)
    scroll = Scroll(location=(scene.maze, (5, 5)), scene=scene)

    for _ in range(3):
        scene.update_sight(1/60.)
    assert scene.sight.version == 3
    assert scene.sight.data.max() > 0

    # full game-side dialog pipeline, headless: reading the scroll opens a
    # pager (its own thread + a grid in scene.grids); it is not consumed
    InputDispatcher.reset()
    dispatcher = InputDispatcher()
    session = player.read(scroll)
    assert dispatcher.handlers[-1] is session
    assert len(scene.grids) == 1
    session.post(KeyPress('Escape'))
    session.join(10)
    assert session.finished.is_set() and session.error is None
    assert scroll in scene.items                 # reading no longer destroys it
    assert scroll in scene.maze.inventory[(5, 5)]
    assert len(scene.grids) == 0

    forbidden = [m for m in sys.modules if m.split('.')[0] in ('vispy', 'PyQt5', 'OpenGL', 'qtpy', 'PySide2')]
    assert not forbidden, "rendering modules were imported: %s" % forbidden
    print("OK: game model ran headless; no rendering module imported")


if __name__ == '__main__':
    main()
