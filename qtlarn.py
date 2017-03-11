import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np


class Text(QtGui.QGraphicsTextItem):
    def __init__(self):
        QtGui.QGraphicsTextItem.__init__(self)
        font = pg.QtGui.QFont('monospace')
        self.setFont(font)

    def setHtml(self, html):
        QtGui.QGraphicsTextItem.setHtml(self, html)
        self.resetTransform()
        br = self.boundingRect()
        fm = QtGui.QFontMetrics(self.font())
        w = fm.averageCharWidth()
        h = fm.height()
        print w, h
        self.scale(1.0/w, 1.0/h)


class Console(Text):
    def __init__(self, shape):
        self.textshape = shape
        char_template = b'<span style="color:#FFFFFF;background:#000000">#</span>'
        row_template = (char_template * shape[1]) + b'<br/>\n'
        self.html = bytearray(row_template * shape[0])
        self.html_array = np.frombuffer(self.html, count=shape[0] * len(row_template), dtype='S1').reshape(shape[0], len(row_template))
        self.all_chars_array = self.html_array[:, :-6].reshape(shape[0], shape[1], len(char_template))
        self.color_array = self.all_chars_array[:, :, 20:26]
        self.char_array = self.all_chars_array[:, :, -8]
        
        Text.__init__(self)
        font = pg.QtGui.QFont('monospace')
        self.setFont(font)
        self.update_all()
        
    def update_all(self):
        self.setHtml(str(self.html))
        


class Character(Text):
    def __init__(self, char="@"):
        Text.__init__(self)
        self.setHtml('<span style="color:#000000;background:#CCCCCC">%s</span>' % char)
        


if __name__ == '__main__':
    app = pg.mkQApp()
    w = pg.GraphicsLayoutWidget()
    v = w.addViewBox()
    v.invertY()
    v.setAspectLocked()
    w.show()
    w.resize(1200, 800)

    shape = (50, 150)

    c = Console(shape)
    c.setZValue(-1)
    v.addItem(c)

    maze = np.zeros(shape, dtype=int)
    maze[0] = 1
    maze[-1] = 1
    maze[:,0] = 1
    maze[:,-1] = 1
    
    chars = np.array([".", "#"])[maze]
    c.char_array[:] = chars
    c.update_all()
    
    player = Character("@")
    player.setPos(3, 3)
    v.addItem(player)
    