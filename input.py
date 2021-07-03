import sys, re
import inputs
import threading
from PyQt5 import QtGui, QtCore
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
        for handler in self.handlers[::-1]:
            if handler not in self.handlers:
                continue
            if handler.gamepad_event(event) is True:
                break        

    def key_pressed(self, event):
        for handler in self.handlers[::-1]:
            if handler not in self.handlers:
                continue
            if handler.key_pressed(event) is True:
                break        

    def key_released(self, event):
        for handler in self.handlers[::-1]:
            if handler not in self.handlers:
                continue
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
        if not self.active:
            disp.add_handler(self)

    def deactivate(self):
        InputDispatcher.dispatcher.remove_handler(self)

    @property
    def active(self):
        disp = InputDispatcher.dispatcher
        return self in disp.handlers
    
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
        elif ev.key == 'Tab':
            self.scene.toggle_command_mode()
        elif ev.key == 't':
            self.scene.command('take')
        elif ev.key == 'd':
            self.scene.command('drop')
        elif ev.key == 'r':
            self.scene.command('read')
        else:
            self.keys.add(ev.key)
            self.handle_input(None)
        
    def key_released(self, ev):
        try:
            self.keys.remove(ev.key)
        except KeyError:
            pass


class CommandInputHandler(InputHandler):
    def __init__(self, console, interpreter):
        self.console = console
        self.interpreter = interpreter
        InputHandler.__init__(self)
        self.command = ""
        self.command_history = []
        self.console_line_started = False

    def activate(self):
        InputHandler.activate(self)
        self.update_prompt()

    def deactivate(self):
        InputHandler.deactivate(self)
        self.clear_prompt()

    def key_pressed(self, ev):
        if ev.key in ['Escape', 'Tab']:
            # pass on to default handler
            return False

        if ev.key == 'Enter':
            self.run_command()
            return True
        
        # todo: command history up/down
        # todo: cursor left/right/bkspc/del/home/end

        s = ev.text
        self.command += ev.text
        self.update_prompt()
        
        return True

    def update_prompt(self, cursor=True):
        line = "> " + self.command
        if cursor:
            line += '_'
        if not self.console_line_started:
            self.console.write("\n" + line)
            self.console_line_started = True
        else:
            self.console.set_last_line(line)

    def clear_prompt(self):
        self.command = ""
        if self.console_line_started:
            self.console.remove_last_line()
            self.console.remove_last_line()
            self.console_line_started = False

    def run_command(self):
        cmd = self.command
        self.clear_prompt()
        self.command_history.append(cmd)
        self.interpreter(cmd)
        if self.active:
            self.update_prompt()


class MenuInputHandler(InputHandler):
    """For handling input during normal (non-command) play when a menu of items is presented to the user.
    """
    def __init__(self, interpreter):
        self.interpreter = interpreter
        InputHandler.__init__(self)
        
    def key_pressed(self, ev):
        if re.match(r'[a-zA-Z]$', ev.text) is not None:
            self.interpreter(ev.text)
        else:
            self.interpreter.cancel()

        self.deactivate()        
        return True


class InputThread(QtCore.QThread):
    """Thread for polling joystick input
    """
    new_event = QtCore.pyqtSignal(object, object)
    
    def __init__(self, device=None):
        QtCore.QThread.__init__(self)
        if device is None:
            gp = inputs.devices.gamepads
            if len(gp) == 0:
                print("No gamepads found.")
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
