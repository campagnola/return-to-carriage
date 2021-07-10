# coding: utf8
try:
    import faulthandler
    faulthandler.enable()
except ImportError:
    pass

import vispy.scene, vispy.app

from carriage_return.scene import Scene
from carriage_return.monster import Monster
from carriage_return.item import Scroll, Torch


if __name__ == '__main__':
    canvas = vispy.scene.SceneCanvas()
    canvas.show()
    canvas.size = 1400, 900
    
    scene = Scene(canvas)

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
    
