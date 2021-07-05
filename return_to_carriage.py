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

    scroll = Scroll(location=(scene, (5, 5)), scene=scene)
    torches = [
        Torch(location=(scene, (17, 8)), scene=scene),
        Torch(location=(scene, (3, 8)), scene=scene),
        Torch(location=(scene, (9,  30)), scene=scene),
        Torch(location=(scene, (32, 41)), scene=scene),
        Torch(location=(scene, (32, 45)), scene=scene),
        Torch(location=(scene, (43, 39)), scene=scene),

        Torch(location=(scene, (16, 75)), scene=scene),
        Torch(location=(scene, (28, 75)), scene=scene),
        Torch(location=(scene, (40, 75)), scene=scene),
        Torch(location=(scene, (52, 75)), scene=scene),
        Torch(location=(scene, (16, 82)), scene=scene),
        Torch(location=(scene, (28, 82)), scene=scene),
        Torch(location=(scene, (40, 82)), scene=scene),
        Torch(location=(scene, (52, 82)), scene=scene),
    ]
    yeti = Monster(position=(8, 40), scene=scene)
    
