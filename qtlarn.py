import pyqtgraph as pg
import numpy as np

app = pg.mkQApp()
w = pg.GraphicsLayoutWidget()
v = w.addViewBox()
v.invertY()
v.setAspectLocked()
w.show()

#class Map(pg.QtGui.QGraphicsItem):
    #def __init__(self):
        #self.text = 

item = pg.QtGui.QGraphicsTextItem("line1\nline2")
font = pg.QtGui.QFont('monospace')
item.setFont(font)
v.addItem(item)

#bg = pg.ImageItem()
#v.addITem(bg)
#bg.setZValue(-1)

def update():
    map = np.random.randint(0, 2, size=(50, 140))
    map[:,-1] = 2
    chars = np.array(list('.#\n'))
    text = chars[map].tostring()
    item.setPlainText(text)

timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(16)

