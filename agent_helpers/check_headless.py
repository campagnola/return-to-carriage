"""Prove the game model runs with no rendering library loaded.

Builds a Scene with a fake numpy visibility provider, runs sight updates,
exercises the message channel and item destruction, then asserts that no
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

    # message channel + item destruction
    messages = []
    scene.messages.connect(lambda event: messages.append(event.message))
    scroll.read(player)
    assert len(messages) == 1 and 'scroll' in messages[0]
    assert scroll not in scene.items
    assert scroll not in scene.maze.inventory[(5, 5)]
    assert np.isnan(scroll.sprite.sprite.position).all()

    forbidden = [m for m in sys.modules if m.split('.')[0] in ('vispy', 'PyQt5', 'OpenGL', 'qtpy', 'PySide2')]
    assert not forbidden, "rendering modules were imported: %s" % forbidden
    print("OK: game model ran headless; no rendering module imported")


if __name__ == '__main__':
    main()
