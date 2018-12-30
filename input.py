import sys
import inputs
import threading
from PyQt4 import QtGui, QtCore
import vispy.app
import vispy.util.ptime as ptime


class InputDispatcher(object):
    """Receives all user input events and dispatches them to event handlers.
    """
    dispatcher = None

    def __init__(self, canvas):
        if InputDispatcher.dispatcher is not None:
            raise Exception("Only one InputDispatcher allowed.")
        InputDispatcher.dispatcher = self

        self.handlers = []

        # get keyboard events from the vispy canvas
        self._canvas = canvas
        canvas.events.key_press.connect(self.key_pressed)
        canvas.events.key_release.connect(self.key_released)

        # get gamepad events from a background thread
        try:
            self.gamepad_thread = InputThread()
            self.gamepad_thread.new_event.connect(self.gamepad_event)
        except Exception:
            self.gamepad_thread = None
            sys.excepthook(*sys.exc_info())

    def add_handler(self, handler):
        """Add a new input handler to the beginning of the list of input handlers.

        The new handler will be the first to receive all user input until it is
        removed from the stack or another handler is added on top of it.
        """
        self.handlers.append(handler)
        
    def remove_handler(self, handler):
        """Remove the first instance of a handler from the list of input handlers.

        The handler will no longer receive input events unless it existed multiple times
        in the handler list.
        """
        self.handlers.remove(handler)

    def gamepad_event(self, event):
        for handler in self.handlers:
            if handler.gamepad_event(event) is True:
                break        

    def key_pressed(self, event):
        for handler in self.handlers:
            if handler.key_pressed(event) is True:
                break        

    def key_released(self, event):
        for handler in self.handlers:
            if handler.key_released(event) is True:
                break        


class InputHandler(object):
    """Base class for user input handling.

    Each InputHandler subclass is responsible for handling user events in a
    particular context (for example, keyboard event handling changes when we ask the user
    a question).
    """
    def __init__(self):
        self.gamepad_state = {}
        self.keys = set()

    def activate(self):
        disp = InputDispatcher.dispatcher
        if self not in disp.handlers:
            disp.add_handler(self)

    def deactivate(self):
        InputDispatcher.dispatcher.remove_handler(self)
    
    def gamepad_event(self, event, state):
        """Called when the gamepad state has changed.

        Return True if this event is handled, False to allow downstream handlers
        to receive the event.
        """
        return True

    def key_presssed(self, event):
        """Called when a key is pressed.

        Return True if this event is handled, False to allow downstream handlers
        to receive the event.
        """
        return True

    def key_released(self, event):
        """Called when a key is released.

        Return True if this event is handled, False to allow downstream handlers
        to receive the event.
        """
        return True


class DefaultInputHandler(InputHandler):
    """Gamepad and keyboard handling during normal gameplay.
    """
    def __init__(self, scene):
        self.scene = scene
        InputHandler.__init__(self)

        self.last_input_update = ptime.time()
        self.input_timer = vispy.app.Timer(start=True, connect=self.handle_input, interval=0.016)

    def gamepad_event(self, ev, state):
        # gamepad input
        self.gamepad_state = state
        self.handle_input(None)
 
    def handle_input(self, ev):
        now = ptime.time()
        dt = now - self.last_input_update
        
        gp = self.gamepad_state
        gp_south = gp.get('BTN_SOUTH', 0) == 1
        wait = 0.05 if ('Shift' in self.keys or gp_south) else 0.1
        if dt < wait:
            return

        gp_x = gp.get('ABS_HAT0X', 0)
        gp_y = gp.get('ABS_HAT0Y', 0)
        dx = [gp_x, -gp_y]
        if 'Right' in self.keys:
            dx[0] += 1
        if 'Left' in self.keys:
            dx[0] -= 1
        if 'Up' in self.keys:
            dx[1] += 1
        if 'Down' in self.keys:
            dx[1] -= 1
        
        if dx[0] == 0 and dx[1] == 0:
            return
        
        pos = self.scene.player.position
        j0, i0 = pos.astype('uint')
        newpos = pos + dx
        
        self.scene.request_player_move(newpos.astype('uint'))
        
        self.last_input_update = now
    
    def key_pressed(self, ev):
        if ev.key == 'Escape':
            self.scene.quit()
        if ev.key == 't':
            self.scene.request_player_action('take')
        elif ev.key == 'r':
            self.scene.request_player_action('read')
        
        self.keys.add(ev.key)
        self.handle_input(None)
        
    def key_released(self, ev):
        try:
            self.keys.remove(ev.key)
        except KeyError:
            pass


class KeyInputHandler(InputHandler):
    """For handling exclusively keyboard input (such as when entering text)

    Optional callback is invoked for every press or release event.
    """
    def __init__(self, callback=None):
        self.callback = callback
        self.text = ""
        self.key_events = []
        self.press_events = []
    
    def key_pressed(self, event):
        self.text = self.text + event.text
        self.key_events.append(event)
        self.press_events.append(event)
        if self.callback is not None:
            self.callback(self, event)

    def key_released(self, event):
        self.key_events.append(event)
        if self.callback is not None:
            self.callback(self, event)


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


    