"""Deterministically build the game scene and save a screenshot PNG.

Used as a rendering regression check: capture a baseline once, then re-run
after any change and compare with compare_screenshots.py. Two baselines are
kept (see ARCHITECTURE.md's Verification section): the base frame, and a
frame with the take menu open (``--menu``), which exercises the generic
grid renderer (backends/vispy/grids.py) end-to-end.

Usage: python agent_helpers/render_screenshot.py <output.png> [--menu]
"""
import os
import sys

import numpy as np

np.random.seed(12345)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)  # level1.png is loaded from cwd

from carriage_return.backends.vispy import MainWindow, VispySceneRenderer
from carriage_return.scene import Scene
from carriage_return.dm import DungeonMaster
from carriage_return.hud import build_hud
from carriage_return.input import InputDispatcher
from carriage_return.game import new_game
from carriage_return.item import Scroll, Torch


def build_game():
    dispatcher = InputDispatcher()
    ui = MainWindow(dispatcher)
    scene = Scene()
    hud = build_hud(scene)
    scene.write('Hello?')
    scene.write('Is anybody\n    there?')
    renderer = VispySceneRenderer(ui, scene)
    ui.attach_scene(scene)
    dm = DungeonMaster(scene)

    # note: no ui.follow_entity() and no GameplayInputHandler -- camera
    # scrolling and the input threads are timing-dependent.
    # Fewer torches than the main game on purpose: torch count and placement
    # change the lighting, so this list is part of the screenshot baseline.
    player = new_game(scene, torch_positions=[(17, 8), (3, 8), (9, 30)])[0]

    return ui, scene, renderer, player


def open_take_menu(scene, player):
    """Put two items under the player and open the take menu.

    ``dialogs.open_menu`` builds the MenuPainter and paints it synchronously
    on the calling thread (only the dialog's key loop runs on its own daemon
    thread), so the menu's grid is present in scene.grids as soon as this
    call returns -- no GUI event pump or timing dependency needed for a
    deterministic "menu open" screenshot.
    """
    from carriage_return.interpreter import CommandInterpreter
    pos = player.location.slot
    Scroll(location=(scene.maze, pos), scene=scene)
    Torch(location=(scene.maze, pos), scene=scene)
    CommandInterpreter(scene).take([])  # ambiguous (2 items) -> opens a menu


def main():
    out_path = sys.argv[1]
    menu = len(sys.argv) > 2 and sys.argv[2] == '--menu'
    ui, scene, renderer, player = build_game()
    if menu:
        open_take_menu(scene, player)

    # SceneCanvas.render() draws the scene directly without emitting
    # events.draw or running the frame tick, so the renderer's per-draw
    # update (LOS/lighting -> sight texture) and the grid sync must be
    # invoked explicitly; a fixed dt keeps the memory decay deterministic.
    # Two rounds reach steady state.
    ui.canvas.set_current()
    for _ in range(2):
        renderer.update(dt=1/60.)
        ui.grid_renderer.sync()
        img = ui.canvas.render()

    import PIL.Image
    # drop the alpha channel: the on-screen window ignores framebuffer
    # alpha, so RGB is the faithful record (alpha < 255 where translucent
    # cell backgrounds drew would white-bleed in image viewers)
    PIL.Image.fromarray(img[..., :3]).save(out_path)
    print("saved %s shape=%s" % (out_path, img.shape))


if __name__ == '__main__':
    main()
