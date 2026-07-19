# coding: utf8
import faulthandler
faulthandler.enable()

from carriage_return.backends.vispy import MainWindow, VispySceneRenderer
from carriage_return.scene import Scene
from carriage_return.dm import DungeonMaster
from carriage_return.hud import build_hud
from carriage_return.player import Player
from carriage_return.monster import Monster
from carriage_return.item import Scroll, Torch
from carriage_return.interpreter import CommandInterpreter
from carriage_return.input import (InputDispatcher, GameplayInputHandler,
                                   CommandInputHandler, start_gamepad)

import sys
import vispy.app



if __name__ == '__main__':

    dispatcher = InputDispatcher()
    ui = MainWindow(dispatcher)
    scene = Scene()
    hud = build_hud(scene)
    scene.write('Hello?')
    scene.write('Is anybody\n    there?')
    renderer = VispySceneRenderer(ui, scene)
    ui.attach_scene(scene)
    dm = DungeonMaster(scene)

    player = Player(scene)
    player.location.update(scene.maze, [7, 7])
    ui.follow_entity(player)

    interp = CommandInterpreter(scene)
    cmd_input_handler = CommandInputHandler(scene.log, interp)
    gameplay = GameplayInputHandler(dm, player, interpreter=interp,
                                    command_handler=cmd_input_handler)
    gameplay.activate()
    start_gamepad(dispatcher)

    scroll = Scroll(location=(scene.maze, (5, 5)), scene=scene)
    torches = [
        Torch(location=(scene.maze, (17, 8)), scene=scene),
        Torch(location=(scene.maze, (3, 8)), scene=scene),
        Torch(location=(scene.maze, (9,  30)), scene=scene),
        Torch(location=(scene.maze, (32, 41)), scene=scene),
        Torch(location=(scene.maze, (32, 45)), scene=scene),
        Torch(location=(scene.maze, (43, 39)), scene=scene),

        Torch(location=(scene.maze, (15, 75)), scene=scene),
        Torch(location=(scene.maze, (28, 75)), scene=scene),
        Torch(location=(scene.maze, (40, 75)), scene=scene),
        Torch(location=(scene.maze, (52, 75)), scene=scene),
        Torch(location=(scene.maze, (15, 82)), scene=scene),
        Torch(location=(scene.maze, (28, 82)), scene=scene),
        Torch(location=(scene.maze, (40, 82)), scene=scene),
        Torch(location=(scene.maze, (52, 82)), scene=scene),
    ]
    torches[0].light_color = (10000, 5000, 1000)

    held_torch = Torch(location=(scene.player, 'right hand'), scene=scene, obj_name="held torch")
    held_torch.light_color = (10000, 5000, 1000)

    yeti = Monster(position=(8, 40), scene=scene)
    
    if sys.flags.interactive == 0:
        vispy.app.run()