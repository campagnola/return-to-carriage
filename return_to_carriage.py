# coding: utf8
try:
    import faulthandler
    faulthandler.enable()
except ImportError:
    pass

from PyQt5 import QtGui, QtCore
import vispy.scene, vispy.app

from scene import Scene
from monster import Monster
from item import Item


if __name__ == '__main__':
    canvas = vispy.scene.SceneCanvas()
    canvas.show()
    canvas.size = 1400,900
    
    scene = Scene(canvas)

    scroll = Item(location=(scene, (5, 5)), scene=scene)
    yeti = Monster(position=(8, 40), scene=scene)
    
