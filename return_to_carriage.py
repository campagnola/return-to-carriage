# coding: utf8
import faulthandler
faulthandler.enable()

from carriage_return.ui import MainWindow
from carriage_return.scene import Scene
from carriage_return.dm import DungeonMaster
from carriage_return.player import Player
from carriage_return.monster import Monster
from carriage_return.item import Scroll, Torch
from carriage_return.input import DefaultInputHandler


if __name__ == '__main__':

    ui = MainWindow()
    scene = Scene(ui)
    dm = DungeonMaster(scene)

    player = Player(scene)
    player.location.update(scene.maze, [7, 7])

    default_input_handler = DefaultInputHandler(dm, ui, player)
    ui.input_dispatcher.add_handler(default_input_handler)

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

    held_torch = Torch(location=(scene.player, 'right hand'), scene=scene)
    held_torch.light_color = (10000, 5000, 1000)

    yeti = Monster(position=(8, 40), scene=scene)
    
