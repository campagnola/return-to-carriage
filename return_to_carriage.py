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
    torch = Torch(location=(scene, (17, 12)), scene=scene)
    yeti = Monster(position=(8, 40), scene=scene)
    
