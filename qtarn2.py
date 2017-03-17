import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np


class Text1(QtGui.QGraphicsItem):
    def __init__(self, shape=(50, 120)):
        QtGui.QGraphicsItem.__init__(self)
        self.brushes = []
        self.size = shape
        self.txt = np.array(['#',  '.'])[np.random.randint(0, 2, size=self.size)]
        for i in range(self.size[0]):
            row = []
            for j in range(self.size[1]):
                row.append(pg.mkPen(*np.random.randint(0, 255, size=3)))
            self.brushes.append(row)

    def randomize(self):
        self.txt = np.array(['#',  '.'])[np.random.randint(0, 2, size=self.size)]
        self.brushes = []
        for i in range(self.size[0]):
            row = []
            for j in range(self.size[1]):
                row.append(pg.mkPen(*np.random.randint(0, 255, size=3)))
            self.brushes.append(row)
        self.update()

    def boundingRect(self):
        return pg.QtCore.QRectF(0, 0, self.size[1]*10, self.size[0]*10)

    def paint(self, p, *args):
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                p.setPen(self.brushes[i][j])
                p.drawText(pg.QtCore.QPointF(j*10, i*10), self.txt[i,j])


class Text2(QtGui.QGraphicsItemGroup):
    def __init__(self, shape=(50, 120)):
        QtGui.QGraphicsItemGroup.__init__(self)
        self.chars = []
        self.size = shape
        for i in range(self.size[0]):
            self.chars.append([])
            for j in range(self.size[1]):
                char = QtGui.QGraphicsTextItem("#", self)
                char.setDefaultTextColor(pg.mkColor(*np.random.randint(0, 255, size=3)))
                char.setPos(j*10, i*10)
                char.setCacheMode(char.DeviceCoordinateCache)
                self.chars[-1].append(char)
        

class Text3(QtGui.QGraphicsItemGroup):
    def __init__(self, shape=(50, 120)):
        QtGui.QGraphicsItemGroup.__init__(self)
        
        chars = np.array(['#',  '.'])
        paths = {}
        for char in chars:
            path = QtGui.QPainterPath()
            path.addText(0, 0, QtGui.QFont("monospace", 10), char)
            #br = path.boundingRect()
            #scale = min(1. / br.width(), 1. / br.height())
            scale = 0.1
            tr = QtGui.QTransform()
            tr.scale(scale, scale)
            #tr.translate(-br.x() - br.width()/2., -br.y() - br.height()/2.)
            path = tr.map(path)
            paths[char] = path

        txt = chars[np.random.randint(0, 2, size=shape)]
        self.s = pg.ScatterPlotItem(pxMode=True, antialias=True)   ## Set pxMode=False to allow spots to transform with the view
        self.s.setParentItem(self)
        spots = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                char = txt[i, j]
                spots.append({'pos': (j*10, i*10), 'size': 10, 'pen': None, 'brush': pg.mkBrush(*np.random.randint(0, 255, size=3)), 'symbol':paths[char]})

        self.s.addPoints(spots)




class Text4(QtGui.QGraphicsItem):
    def __init__(self, shape=(50, 120)):
        QtGui.QGraphicsItem.__init__(self)
        chars = np.array(['#',  '.'])
        self.font = QtGui.QFont('monospace', 16)
        fm = QtGui.QFontMetrics(self.font)
        charsize = (int(fm.height()), int(fm.maxWidth()))
        self.glyphs = np.empty(chars.shape + charsize + (3,), dtype='ubyte')
        for i,char in enumerate(chars):
            img = QtGui.QImage(charsize[1], charsize[0], QtGui.QImage.Format_RGB32)
            p = QtGui.QPainter()
            p.begin(img)
            p.fillRect(0, 0, charsize[1], charsize[0], pg.mkBrush('r'))
            p.setPen(pg.mkPen('g'))
            p.setFont(self.font)
            p.drawText(0, fm.ascent(), char)
            p.end()
            self.glyphs[i] = pg.imageToArray(img)[..., :3].transpose(1, 0, 2)
        
        self.size = shape
        self.map = np.random.randint(0, 2, size=self.size)
        self.fgcolor = np.random.randint(0, 256, size=(self.size + (3,)), dtype='ubyte')
        self.bgcolor = np.random.randint(0, 256, size=(self.size + (3,)), dtype='ubyte')
        self.img = None

    def randomize(self):
        p = pg.debug.Profiler(disabled=True)
        self.map = np.random.randint(0, 2, size=self.size)
        self.fgcolor = np.random.randint(0, 256, size=(self.size + (3,)), dtype='ubyte')
        self.bgcolor = np.random.randint(100, 120, size=(self.size + (3,)), dtype='ubyte')
        self.img = None
        p()
        self.update()
        p()
        
    def get_image(self):
        p = pg.debug.Profiler(disabled=True)
        if self.img is None:
            gs = self.glyphs.shape[1:3]
            ms = self.size
            img = self.glyphs.astype('uint16')[self.map]
            p('glyphs[map]')
            img = img[...,1:2] * self.fgcolor[:, :, None, None, :] + img[...,2:3] * self.bgcolor[:, :, None, None, :]
            p('recolor')
            img = (img // 256).astype('ubyte')
            p('rescale')
            img = img.transpose(0, 2, 1, 3, 4).reshape(ms[0]*gs[0], ms[1]*gs[1], 3)
            p('reshape')
            self.img = pg.makeQImage(img, transpose=False, copy=True)
            p('qimage')
        return self.img
        
    def boundingRect(self):
        gs = self.glyphs.shape[1:3]
        ms = self.size
        return QtCore.QRectF(0, 0, gs[1]*ms[1], gs[0]*ms[0])
    
    def paint(self, p, *args):
        prof = pg.debug.Profiler(disabled=True)
        br = self.boundingRect()
        img = self.get_image()
        prof("get image")
        p.drawImage(br, img, br)
        prof("draw image")





class BaseText(QtGui.QGraphicsTextItem):
    def __init__(self):
        QtGui.QGraphicsTextItem.__init__(self)
        font = pg.QtGui.QFont('monospace')
        self.setFont(font)

    #def setHtml(self, html):
        #QtGui.QGraphicsTextItem.setHtml(self, html)
        #self.resetTransform()
        #br = self.boundingRect()
        #fm = QtGui.QFontMetrics(self.font())
        #w = fm.averageCharWidth()
        #h = fm.height()
        #self.scale(1.0/w, 1.0/h)


class Text5(BaseText):
    def __init__(self, shape=(50, 120)):
        self.size = shape
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
        
        BaseText.__init__(self)
        font = pg.QtGui.QFont('monospace')
        self.setFont(font)
        self.update_all()

    def randomize(self):
        map = np.random.randint(0, 2, size=self.size)
        chars = np.array(['#',  '.'])
        self.char_array[:] = chars[map]
        self.color = np.random.randint(0, 256, size=(self.size + (3,)), dtype='ubyte')
        self.bgcolor = np.random.randint(0, 256, size=(self.size + (3,)), dtype='ubyte')
        self.update_all()
        
    def update_all(self):
        hex = np.array(list('0123456789ABCDEF'))
        self.bgcolor_array[...,::2] = hex[self.bgcolor>>4]
        self.bgcolor_array[...,1::2] = hex[self.bgcolor&0xF]
        self.color_array[...,::2] = hex[self.color>>4]
        self.color_array[...,1::2] = hex[self.color&0xF]
        self.setHtml(str(self.html))









if __name__ == '__main__':
    app = pg.mkQApp()
    #w = pg.GraphicsLayoutWidget()
    #v = w.addViewBox(0, 0)
    #v.invertY()
    w = pg.GraphicsView()
    txt = Text4(shape=(40, 120))
    w.scene().addItem(txt)
    br = txt.boundingRect()
    w.resize(br.width(), br.height())
    #v.addItem(txt)
    w.show()

    import time
    dt = 0.0
    last = time.time()
    def update():
        global last, dt
        txt.randomize()
        now = time.time()
        dt = dt * 0.9 + (now-last) * 0.1 
        last = now
        print 1.0/dt
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(16)


