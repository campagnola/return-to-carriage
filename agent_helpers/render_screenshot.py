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

from offscreen_clock import ManualClock  # tooling only; see module docstring


def build_game():
    dispatcher = InputDispatcher()
    ui = MainWindow(dispatcher)
    scene = Scene()
    hud = build_hud(scene)
    scene.write('Hello?')
    scene.write('Is anybody\n    there?')
    # deterministic time source so adaptation/memory decay do not depend on how
    # fast this process runs (offscreen tooling only; see offscreen_clock).
    clock = ManualClock()
    renderer = VispySceneRenderer(ui, scene, time_source=clock)
    ui.attach_scene(scene)
    dm = DungeonMaster(scene)

    # note: no ui.follow_entity() and no GameplayInputHandler -- camera
    # scrolling and the input threads are timing-dependent.
    world, player = new_game(scene)

    return ui, scene, renderer, player, clock


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
    ui, scene, renderer, player, clock = build_game()
    if menu:
        open_take_menu(scene, player)

    # SceneCanvas.render() draws the scene (recomputing the LOS/lighting fields
    # in the sprite pre-draw) but does not run the frame tick, so the grid sync
    # is invoked explicitly. The first render establishes the clock baseline
    # (dt == 0); two further frames advance a fixed 1/60 s each so memory decay
    # and adaptation are deterministic, reaching the same steady state every run.
    ui.canvas.set_current()
    ui.grid_renderer.sync()
    ui.canvas.render()                      # warm-up: dt == 0
    for _ in range(2):
        clock.advance(1 / 60.)
        ui.grid_renderer.sync()
        img = ui.canvas.render()            # dt == 1/60

    import PIL.Image
    # drop the alpha channel: the on-screen window ignores framebuffer
    # alpha, so RGB is the faithful record (alpha < 255 where translucent
    # cell backgrounds drew would white-bleed in image viewers)
    PIL.Image.fromarray(img[..., :3]).save(out_path)
    print("saved %s shape=%s" % (out_path, img.shape))


if __name__ == '__main__':
    main()
