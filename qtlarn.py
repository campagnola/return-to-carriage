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
        self.scale(1.0/w, 1.0/h)


class CharGrid(Text):
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
    def __init__(self, char):
        Text.__init__(self)
        self.setHtml('<span style="color:#000000;background:#CCCCCC">%s</span>' % char)
        

class MainWindow(QtGui.QWidget):
    key_pressed = QtCore.Signal(object)
    key_released = QtCore.Signal(object)
    
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self._layout = QtGui.QGridLayout()
        self.setLayout(self._layout)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.gw = pg.GraphicsLayoutWidget()
        self._layout.addWidget(self.gw)
        self.v = self.gw.addViewBox()
        self.v.invertY()
        self.v.setAspectLocked()
        self.show()
        self.resize(1200, 800)

    def keyPressEvent(self, ev):
        self.key_pressed.emit(ev)
        
    def keyReleaseEvent(self, ev):
        self.key_released.emit(ev)


class Game(QtCore.QObject):
    def __init__(self, win):
        QtCore.QObject.__init__(self)
        self.win = win
        win.key_pressed.connect(self.key_pressed)
        win.key_released.connect(self.key_released)
        self.view = win.v

        shape = (50, 150)

        self.c = CharGrid(shape)
        self.c.setZValue(-1)
        self.view.addItem(self.c)

        self.maze = np.zeros(shape, dtype=int)
        self.maze[0] = 1
        self.maze[-1] = 1
        self.maze[:,0] = 1
        self.maze[:,-1] = 1
        
        chars = np.array([".", "#"])[self.maze]
        self.c.char_array[:] = chars
        self.c.update_all()
        
        self.player = Character("@")
        self.player.setPos(3, 3)
        self.view.addItem(self.player)
        
    def key_pressed(self, ev):
        if ev.key() == QtCore.Qt.Key_Left:
            self.move_player(-1, 0)
        elif ev.key() == QtCore.Qt.Key_Right:
            self.move_player(1, 0)
        elif ev.key() == QtCore.Qt.Key_Up:
            self.move_player(0, -1)
        elif ev.key() == QtCore.Qt.Key_Down:
            self.move_player(0, 1)

    def key_released(self, ev):
        #print ev
        pass

    def move_player(self, dx, dy):
        pos = self.player.pos()
        newpos = pos + QtCore.QPoint(dx, dy)
        if self.maze[newpos.y(), newpos.x()] == 0:
            self.player.setPos(newpos)


if __name__ == '__main__':
    app = pg.mkQApp()
    
    w = MainWindow()
    g = Game(w)

    