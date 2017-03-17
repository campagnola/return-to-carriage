from PyQt4 import QtGui, QtCore
from vispy import scene
from vispy.scene.visuals import Text


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


class MainWindow(QtGui.QWidget):
    key_pressed = QtCore.Signal(object)
    key_released = QtCore.Signal(object)
    
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self._layout = QtGui.QGridLayout()
        self.setLayout(self._layout)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = scene.SceneCanvas()
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'panzoom'
        self._layout.addWidget(self.view._native)
        self.show()
        self.resize(1200, 800)

    def keyPressEvent(self, ev):
        self.key_pressed.emit(ev)
        
    def keyReleaseEvent(self, ev):
        self.key_released.emit(ev)




if __name__ == '__main__':
    
    vispy.app.use()
    app = QtGui.QApplication([])
    
    w = MainWindow()
    g = Game(w)

