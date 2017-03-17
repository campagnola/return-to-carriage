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
        self.bgcolor_array = self.all_chars_array[:, :, 39:45]
        self.char_array = self.all_chars_array[:, :, -8]
        
        self.color = np.ones(shape + (3,), dtype='ubyte') * 255
        self.bgcolor = np.zeros(shape + (3,), dtype='ubyte')
        
        Text.__init__(self)
        font = pg.QtGui.QFont('monospace')
        self.setFont(font)
        self.update_all()
        
    def update_all(self):
        hex = np.array(list('0123456789ABCDEF'))
        self.bgcolor_array[...,::2] = hex[self.bgcolor>>4]
        self.bgcolor_array[...,1::2] = hex[self.bgcolor&0xF]
        self.color_array[...,::2] = hex[self.color>>4]
        self.color_array[...,1::2] = hex[self.color&0xF]
        self.setHtml(str(self.html))


class Character(Text):
    def __init__(self, char):
        Text.__init__(self)
        self.setHtml('<span style="color:#000000;background:#5555CC">%s</span>' % char)
        

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
        #self.v.setAspectLocked()
        self.show()
        self.resize(1200, 800)

    def keyPressEvent(self, ev):
        self.key_pressed.emit(ev)
        
    def keyReleaseEvent(self, ev):
        self.key_released.emit(ev)


class Game(QtCore.QObject):
    def __init__(self, win):
        QtCore.QObject.__init__(self)
        
        self.keys = set()
        
        self.win = win
        win.key_pressed.connect(self.key_pressed)
        win.key_released.connect(self.key_released)
        self.view = win.v

        shape = (40, 100)

        self.c = CharGrid(shape)
        self.c.setZValue(-1)
        self.view.addItem(self.c)

        self.maze = np.ones(shape, dtype=int)
        self.maze[1:10, 1:10] = 0
        self.maze[-10:-1, -10:-1] = 0
        self.maze[20:39, 1:80] = 0
        self.maze[5:30, 6] = 0
        self.maze[35, 5:-5] = 0
        
        colors = np.array([
            [80, 80, 80],   # path
            [0, 0,  0],     # wall
        ], dtype='ubyte')
        self.color = colors[self.maze]
        self.bgcolor = np.zeros(shape + (3,), dtype='ubyte')
        
        # randomize wall color a bit
        rock = np.random.randint(45, 55, size=shape + (1,))
        walls = self.maze == 1
        n_walls = walls.sum()
        self.c.bgcolor[walls] = rock[walls]

        self.memory = np.zeros(shape, dtype=float)

        self._update_text()
        
        self.player = Character("@")
        self.player.setPos(3, 3)
        self.view.addItem(self.player)
        
        self.player_move_timer = QtCore.QTimer()
        self.player_move_timer.timeout.connect(self.update_player_position)

    def _update_text(self):
        chars = np.array([".", "#"])[self.maze]
        self.c.char_array[:] = chars
        imem = np.clip(1.0 / self.memory, 0, 255).astype('ubyte')[...,None]
        self.c.color[:] = self.color // imem
        self.c.bgcolor[:] = self.bgcolor // imem
        self.c.update_all()
        
    def key_pressed(self, ev):
        if ev.isAutoRepeat():
            return
        self.keys.add(ev.key())
        self.update_player_position()
        
    def update_player_position(self):
        dir = [0, 0]
        dirkeys = [
            (QtCore.Qt.Key_Left, [-1, 0]),
            (QtCore.Qt.Key_Right, [1, 0]),
            (QtCore.Qt.Key_Up, [0, -1]),
            (QtCore.Qt.Key_Down, [0, 1]),
        ]
        for key, d in dirkeys:
            if key in self.keys:
                dir[0] += d[0]
                dir[1] += d[1]
        self.move_player(*dir)
        if not self.player_move_timer.isActive():
            self.player_move_timer.start(50)

    def key_released(self, ev):
        if ev.isAutoRepeat():
            return
        self.keys.remove(ev.key())
        if len(self.keys) == 0:
            self.player_move_timer.stop()

    def move_player(self, dx, dy):
        pos = self.player.pos()
        pos = (int(pos.x()) + dx, int(pos.y()) + dy)
        if self.maze[pos[1], pos[0]] == 0:
            self.player.setPos(*pos)
            self.memory[pos[1], pos[0]] = 1
            self._update_text()


if __name__ == '__main__':
    app = pg.mkQApp()
    
    w = MainWindow()
    g = Game(w)

    