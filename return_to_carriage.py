# coding: utf8
import faulthandler
faulthandler.enable()

from carriage_return import config
from carriage_return.backends.vispy import MainWindow, VispySceneRenderer
from carriage_return.scene import Scene
from carriage_return.dm import DungeonMaster
from carriage_return.hud import build_hud
from carriage_return.game import new_game
from carriage_return.interpreter import CommandInterpreter
from carriage_return.input import (InputDispatcher, GameplayInputHandler,
                                   CommandInputHandler, start_gamepad)

import argparse
import sys
import vispy.app



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Return to Carriage')
    parser.add_argument('--test', action='store_true',
                        help='test mode: speed up slow real-time behaviours '
                             '(e.g. eye adaptation) for hands-on testing')
    args = parser.parse_args()
    # Set before anything is built so construction-time reads (eye adaptation
    # time constants, etc.) see it.
    config.test_mode = args.test

    dispatcher = InputDispatcher()
    ui = MainWindow(dispatcher)
    scene = Scene()
    hud = build_hud(scene)
    scene.write('Hello?')
    scene.write('Is anybody\n    there?')
    renderer = VispySceneRenderer(ui, scene)
    ui.attach_scene(scene)
    dm = DungeonMaster(scene)

    world, player = new_game(scene)

    # application wiring -- deliberately not in game.py, since the screenshot
    # harness omits all of it
    ui.follow_entity(player)

    interp = CommandInterpreter(scene)
    cmd_input_handler = CommandInputHandler(scene.log, interp)
    gameplay = GameplayInputHandler(dm, player, interpreter=interp,
                                    command_handler=cmd_input_handler)
    gameplay.activate()
    start_gamepad(dispatcher)

    if sys.flags.interactive == 0:
        vispy.app.run()