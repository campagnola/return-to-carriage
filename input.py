import inputs
import threading
from PyQt4 import QtGui, QtCore


class InputThread(QtCore.QThread):
    
    new_event = QtCore.pyqtSignal(object, object)
    
    def __init__(self, device=None):
        QtCore.QThread.__init__(self)
        if device is None:
            gp = inputs.devices.gamepads
            if len(gp) == 0:
                print "No gamepads found."
                return
            self.dev = gp[0]
        else:
            self.dev = device
            
        self.lock = threading.Lock()
        
        self.start()
        
    def run(self):
        state = {}
        while True:
            for ev in self.dev.read():
                if ev.ev_type not in ['Absolute', 'Key']:
                    continue
                state[ev.code] = ev.state
                self.new_event.emit(ev, state.copy())


    