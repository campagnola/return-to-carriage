"""Deterministically build the game scene and save a screenshot PNG.

Used as a rendering regression check during the graphics/game-state split:
capture a baseline before refactoring, then re-run after each stage and
compare with compare_screenshots.py.

Usage: python agent_helpers/render_screenshot.py <output.png>
"""
import os
import sys

import numpy as np

np.random.seed(12345)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)  # level1.png is loaded from cwd

from carriage_return.ui import MainWindow
from carriage_return.scene import Scene
from carriage_return.render_vispy import VispySceneRenderer
from carriage_return.dm import DungeonMaster
from carriage_return.player import Player
from carriage_return.monster import Monster
from carriage_return.item import Scroll, Torch


def build_game():
    ui = MainWindow()
    scene = Scene()
    renderer = VispySceneRenderer(ui, scene)
    scene.messages.connect(lambda event: ui.console.write(event.message))
    dm = DungeonMaster(scene)

    player = Player(scene)
    player.location.update(scene.maze, [7, 7])
    # note: no ui.follow_entity() and no DefaultInputHandler -- camera
    # scrolling and the gamepad thread are timing-dependent.

    scroll = Scroll(location=(scene.maze, (5, 5)), scene=scene)
    torches = [
        Torch(location=(scene.maze, (17, 8)), scene=scene),
        Torch(location=(scene.maze, (3, 8)), scene=scene),
        Torch(location=(scene.maze, (9, 30)), scene=scene),
    ]
    torches[0].light_color = (10000, 5000, 1000)

    held_torch = Torch(location=(player, 'right hand'), scene=scene, obj_name="held torch")
    held_torch.light_color = (10000, 5000, 1000)

    yeti = Monster(position=(8, 40), scene=scene)
    return ui, scene, renderer, player


def main():
    out_path = sys.argv[1]
    ui, scene, renderer, player = build_game()

    # SceneCanvas.render() draws the scene directly without emitting
    # events.draw, so the renderer's per-draw update (LOS/lighting -> sight
    # texture) must be invoked explicitly; a fixed dt keeps the memory decay
    # deterministic. Two rounds reach steady state.
    ui.canvas.set_current()
    for _ in range(2):
        renderer.update(dt=1/60.)
        img = ui.canvas.render()

    import PIL.Image
    PIL.Image.fromarray(img).save(out_path)
    print("saved %s shape=%s" % (out_path, img.shape))


if __name__ == '__main__':
    main()
