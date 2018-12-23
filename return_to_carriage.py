# coding: utf8
try:
    import faulthandler
    faulthandler.enable()
except ImportError:
    pass

from PyQt4 import QtGui, QtCore
import vispy.scene, vispy.app

from scene import Scene
from monster import Monster
from item import Item


if __name__ == '__main__':
    import user
    
    canvas = vispy.scene.SceneCanvas()
    canvas.show()
    canvas.size = 1400,900
    
    scene = Scene(canvas)

    scroll = Item(location=(5, 5), scene=scene)
    yeti = Monster(position=(8, 40), scene=scene)
    