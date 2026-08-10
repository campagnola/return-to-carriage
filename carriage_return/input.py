"""User input: normalized event classes, the handler stack, and the
gameplay/command input handlers.

This module is game-side — no rendering or GUI library is imported here.
Input *sources* construct InputEvent objects and hand them to
``InputDispatcher.dispatch()``; the vispy keyboard source lives in
backends/vispy/input.py, the gamepad source is GamepadThread below.

Threading contract
------------------
``InputHandler.handle()`` runs on whatever thread the input source lives on
(the GUI thread for keyboard events, the gamepad thread for pads). handle()
must only *route* — enqueue the event or set a flag — and return quickly; it
must never mutate game state. Handlers that do real work subclass
QueuedInputHandler and drain their queue from their own thread; the modal
stack guarantees at most one such thread is receiving input (and therefore
mutating game state) at a time. See dialogs.py for the dialog side of the
same contract.
"""
import queue
import sys
import threading
import time
from collections import namedtuple

import inputs
import numpy as np


class InputEvent(object):
    """Base class for normalized user-input events.

    Events carry plain python data (string key names, dicts) — input sources
    normalize their native event types once, at the boundary. Consumers match
    on class (``isinstance(ev, KeyPress)``), never on type strings.
    """

    def __repr__(self):
        attrs = ' '.join('%s=%r' % (name, val)
                         for name, val in sorted(self.__dict__.items())
                         if val is not None)
        return "<%s%s>" % (self.__class__.__name__, ' ' + attrs if attrs else '')


class KeyPress(InputEvent):
    """A key went down. ``key`` is a plain string name; ``text`` is the typed
    character(s), when any."""

    def __init__(self, key, text=None):
        self.key = key
        self.text = text


class KeyRelease(InputEvent):
    """A key came up. Same fields as KeyPress."""

    def __init__(self, key, text=None):
        self.key = key
        self.text = text


class GamepadEvent(InputEvent):
    """The gamepad state changed. ``state`` maps evdev-style codes
    ('ABS_HAT0X', 'BTN_SOUTH', ...) to their current values."""

    def __init__(self, state):
        self.state = state


class MovementTick(InputEvent):
    """Posted by a ``MovementPacer`` thread onto the gameplay handler's own
    queue: time for another movement step. Carries no data -- the gameplay
    handler recomputes direction from its own live key/gamepad state -- and
    exists so the pacer thread never touches game state itself, only routes,
    like every other input source."""


class FocusIn(InputEvent):
    """Delivered by the dispatcher to a handler that just became the top of
    the stack (the handler above it was removed)."""


class FocusOut(InputEvent):
    """Delivered by the dispatcher to a handler that just lost the top of the
    stack (another handler was pushed above it). Stateful handlers drop
    transient state — held keys — here instead of tracking who is above."""


class Close(InputEvent):
    """Ask a dialog session's loop to unwind; DialogSession.get() raises
    DialogClosed when it sees one."""


class InputDispatcher(object):
    """Routes user input events to a stack of handlers.

    ``dispatch()`` offers each event to the handlers from the top of the
    stack down until one consumes it by returning True. On stack changes the
    dispatcher delivers FocusOut to the handler that lost the top and FocusIn
    to the one that regained it, so a handler never needs to ask who is above
    it.

    add_handler/remove_handler/dispatch may be called from any thread: the
    list operations are atomic, handle() implementations only enqueue, and
    focus events are delivered on the thread performing the stack change.
    """
    dispatcher = None

    def __init__(self):
        if InputDispatcher.dispatcher is not None:
            raise Exception("Only one InputDispatcher allowed.")
        InputDispatcher.dispatcher = self
        self.handlers = []

    @classmethod
    def reset(cls):
        """Forget the current dispatcher singleton so a new one can be created.

        Intended for test code.
        """
        cls.dispatcher = None

    def add_handler(self, handler):
        """Push *handler* onto the top of the stack.

        The new handler is the first to be offered all user input until it is
        removed or another handler is added above it. The handler it
        displaced receives FocusOut.
        """
        old_top = self.handlers[-1] if self.handlers else None
        self.handlers.append(handler)
        if old_top is not None:
            old_top.handle(FocusOut())

    def remove_handler(self, handler):
        """Remove the first instance of *handler* from the stack.

        If the top of the stack changed, the newly exposed top handler
        receives FocusIn.
        """
        was_top = len(self.handlers) > 0 and self.handlers[-1] is handler
        self.handlers.remove(handler)
        if was_top and len(self.handlers) > 0:
            self.handlers[-1].handle(FocusIn())

    def dispatch(self, event):
        """Offer *event* to each handler, top of the stack first.

        Return the handler that consumed it (by returning True), or None if
        the event fell through unconsumed.
        """
        for handler in list(self.handlers)[::-1]:
            if handler.handle(event) is True:
                return handler
        return None


class InputHandler(object):
    """Base class for handlers on the dispatcher stack.

    Subclasses implement ``handle(event) -> bool``: return True to consume
    the event, False to let it continue down the stack. A handler is modal
    simply by returning True for everything (QueuedInputHandler does).

    handle() runs on the input source's thread and must only route (enqueue
    or set a flag) — never mutate game state.
    """

    def activate(self):
        disp = InputDispatcher.dispatcher
        if not self.active:
            disp.add_handler(self)

    def deactivate(self):
        InputDispatcher.dispatcher.remove_handler(self)

    @property
    def active(self):
        disp = InputDispatcher.dispatcher
        return disp is not None and self in disp.handlers

    def handle(self, event):
        raise NotImplementedError()


class QueuedInputHandler(InputHandler):
    """A handler that consumes every event into a queue drained elsewhere.

    The owner's thread drains ``queue`` at its own pace; handle() only
    enqueues, so it is safe to call from any input-source thread. Because
    handle() always returns True, a QueuedInputHandler on top of the stack
    captures all input.
    """

    def __init__(self):
        self.queue = queue.Queue()

    def handle(self, event):
        self.queue.put(event)
        return True


#: A held direction (each axis in {-1, 0, 1}) and the speed it should be
#: walked at. Equality-comparable so a handler can tell whether a recomputed
#: velocity actually changed before forwarding it to the pacer.
Velocity = namedtuple('Velocity', ('direction', 'speed'))


class MovementPacer:
    """Decides, on its own thread, when a held direction produces another
    movement step -- and only that: it never touches the DM or player itself,
    only posts a ``MovementTick`` to *tick_queue* (the owning
    GameplayInputHandler's own event queue), so every actual game-state
    mutation still happens on that single thread, same as any other input
    source.

    Fed by *velocity_queue*, onto which the owning handler pushes a new
    ``Velocity`` each time the held keys/gamepad state make one. A held
    direction ticks every ``1 / speed`` seconds. Starting a hold from a stop
    waits out ``start_delay`` before its first tick -- short enough not to be
    felt, long enough for the second key of a diagonal to arrive -- so a
    diagonal starts diagonal instead of taking a stray orthogonal step in
    whichever direction landed first. Any *other* velocity change -- adding
    or dropping an axis while already moving, or a Shift-driven speed change
    -- ticks immediately: this is what lets a diagonal formed mid-hold reach
    a diagonal-only gap in a single, atomic combined step instead of drifting
    each axis across its own cell boundary independently.

    ``step()`` is the whole state machine as one plain, synchronous method
    (``run()`` just calls it in a loop) so tests can drive pacing without
    threads or sleeping.
    """

    def __init__(self, velocity_queue, tick_queue, start_delay, clock=time.monotonic):
        self.velocity_queue = velocity_queue
        self.tick_queue = tick_queue
        self.start_delay = start_delay
        self.clock = clock
        self.velocity = Velocity((0, 0), 0.0)
        self.hold_start = None  # when the current hold began; None if stopped
        self.stepped = False    # has this hold posted its first tick yet?
        self.next_due = None    # when the next tick is due; None if stopped

    def run(self):
        timeout = None
        while True:
            try:
                velocity = self.velocity_queue.get(timeout=timeout)
            except queue.Empty:
                velocity = None
            timeout = self.step(velocity, self.clock())

    def step(self, velocity, now):
        """Advance by one wakeup (a new *velocity*, or None for a scheduled
        tick); return the delay before the pacer should be woken again
        (None to block until the next velocity change)."""
        if velocity is not None:
            self.velocity = velocity
            if velocity.direction == (0, 0):
                self.hold_start = None
                self.stepped = False
                self.next_due = None
                return None
            if self.hold_start is None:
                self.hold_start = now
                self.stepped = False
                self.next_due = now + self.start_delay
            elif self.stepped:
                self.next_due = now  # already moving: take effect right away

        if self.next_due is None or now < self.next_due:
            return None if self.next_due is None else self.next_due - now

        self.tick_queue.put(MovementTick())
        self.stepped = True
        self.next_due = now + 1.0 / self.velocity.speed
        return 1.0 / self.velocity.speed


class GameplayInputHandler(QueuedInputHandler):
    """Keyboard and gamepad handling during normal gameplay.

    Like a dialog, gameplay runs as a sequential loop on its own daemon
    thread, fed by the event queue: movement keys move the player while held,
    action keys hand off to the scene/interpreter, Tab toggles the command
    prompt, Escape requests quit.

    Movement
    --------
    Direction is the sum of unit vectors of the held arrows (plus the
    gamepad hat), speed depends on Shift / the south gamepad button --
    together a ``Velocity`` recomputed after every key/gamepad event. When it
    changes, it is handed to a ``MovementPacer`` running on its own thread,
    which decides the timing (see its docstring) and posts a ``MovementTick``
    back onto this handler's own queue whenever a step is due. Handling that
    tick -- reading the *current* held direction and asking the DM to move --
    is the only thing that actually mutates game state, and it always
    happens on this thread, so the "one thread mutates state at a time"
    contract described in input.py's module docstring holds even though
    timing now lives elsewhere. ``focused`` guards the same contract against
    a tick that was already in flight when a dialog took the stack: it is
    cleared on FocusOut, so a stray tick that arrives afterward is a no-op.

    While another handler (dialog, command prompt) sits above this one on the
    stack, no events arrive and the held-key state was cleared by FocusOut,
    so the loop simply blocks on its queue — suspension is a consequence of
    the stack, not a decision made here.

    *clock* is injectable and the pacer's ``step()`` is a plain method so
    tests can drive movement timing synchronously, without threads.
    """
    walk_speed = 6.0    # cells / second
    run_speed = 18.0    # ... with Shift / BTN_SOUTH held
    start_delay = 0.04  # grace before a fresh hold's first step, to catch diagonals

    def __init__(self, dm, player, interpreter=None, command_handler=None,
                 clock=time.monotonic, start_thread=True):
        QueuedInputHandler.__init__(self)
        self.dm: 'DungeonMaster' = dm
        self.player = player
        self.interpreter = interpreter
        self.command_handler = command_handler
        self.clock = clock
        self.keys = set()
        self.gamepad_state = {}
        self.velocity = Velocity((0, 0), self.walk_speed)
        self.focused = True
        self.movement_queue = queue.Queue()
        self.pacer = MovementPacer(self.movement_queue, self.queue, self.start_delay, clock)
        self.thread = None
        self.pacer_thread = None
        if start_thread:
            self.thread = threading.Thread(target=self._run, name='gameplay-input',
                                           daemon=True)
            self.thread.start()
            self.pacer_thread = threading.Thread(target=self.pacer.run,
                                                  name='gameplay-movement', daemon=True)
            self.pacer_thread.start()

    def _run(self):
        while True:
            self._process(self.queue.get())

    def _process(self, event):
        """Handle one event -- a real input event, or a pacer MovementTick
        -- then forward the resulting velocity to the pacer if it changed."""
        if isinstance(event, KeyPress):
            self._key_press(event)
        elif isinstance(event, KeyRelease):
            self.keys.discard(event.key)
        elif isinstance(event, FocusOut):
            # a dialog or the command prompt took over; drop transient state
            # so movement does not resume by itself when focus returns
            self.keys.clear()
            self.gamepad_state = {}
            self.focused = False
        elif isinstance(event, FocusIn):
            self.focused = True
        elif isinstance(event, GamepadEvent):
            self.gamepad_state = event.state
        elif isinstance(event, MovementTick):
            self._move()

        velocity = Velocity(self._movement_direction(), self._speed())
        if velocity != self.velocity:
            self.velocity = velocity
            self.movement_queue.put(velocity)

    def _move(self):
        if not self.focused:
            return  # a tick already in flight when a dialog took the stack
        direction = np.array(self._movement_direction())
        if not direction.any():
            return
        pos = np.array(self.player.location.slot)
        self.dm.request_player_move(self.player, pos + direction)

    def _key_press(self, ev):
        scene = self.dm.scene
        if ev.key == 'Escape':
            scene.quit_requested = True
        elif ev.key == 'Tab':
            self._toggle_command_mode()
        elif ev.key == 't':
            self.interpreter.take([])
        elif ev.key == 'r':
            self.interpreter.read([])
        elif ev.key == 'd':
            self.interpreter.drop([])
        elif ev.key == 'c':
            self.interpreter.cast([])
        elif ev.text in ('<', '>'):
            # matched on text, not key: these are shift+comma/period, and the
            # vispy source lowercases single-character key names, so the key
            # name for '>' is unreliable while the text is exactly '>'
            self.dm.use_stairs(self.player, ev.text)
        else:
            self.keys.add(ev.key)

    def _toggle_command_mode(self):
        handler = self.command_handler
        if handler is None:
            return
        if handler.active:
            handler.deactivate()
        else:
            handler.activate()

    def _speed(self):
        """Cells per second for the current walk/run state."""
        gp_south = self.gamepad_state.get('BTN_SOUTH', 0) == 1
        return self.run_speed if ('Shift' in self.keys or gp_south) else self.walk_speed

    def _movement_direction(self):
        gp = self.gamepad_state
        dx = [gp.get('ABS_HAT0X', 0), -gp.get('ABS_HAT0Y', 0)]
        if 'Right' in self.keys:
            dx[0] += 1
        if 'Left' in self.keys:
            dx[0] -= 1
        if 'Up' in self.keys:
            dx[1] += 1
        if 'Down' in self.keys:
            dx[1] -= 1
        return (dx[0], dx[1])


class CommandInputHandler(QueuedInputHandler):
    """Console command-prompt input, on its own thread.

    While active, typed characters build up a command line shown as the last
    line of *log* (a scene MessageLog); Enter runs it through *interpreter*
    on this handler's thread. Escape and Tab are refused (handle() returns
    False) so they fall through to the gameplay handler, which owns toggling
    command mode.
    """

    def __init__(self, log, interpreter, start_thread=True):
        QueuedInputHandler.__init__(self)
        self.log = log
        self.interpreter = interpreter
        self.command = ""
        self.command_history = []
        self.console_line_started = False
        self.thread = None
        if start_thread:
            self.thread = threading.Thread(target=self._run, name='command-input',
                                           daemon=True)
            self.thread.start()

    def activate(self):
        # runs on the thread toggling command mode (the gameplay thread); the
        # prompt draw is a game-state write, legal there
        InputHandler.activate(self)
        self.update_prompt()

    def deactivate(self):
        InputHandler.deactivate(self)
        self.clear_prompt()

    def handle(self, event):
        if isinstance(event, KeyPress) and event.key in ('Escape', 'Tab'):
            # gameplay handler below owns quit / command-mode toggling
            return False
        return QueuedInputHandler.handle(self, event)

    def _run(self):
        while True:
            self._process(self.queue.get())

    def _process(self, event):
        if not isinstance(event, KeyPress):
            return
        if event.key == 'Enter':
            self.run_command()
            return
        # todo: command history up/down
        # todo: cursor left/right/del/home/end
        if event.key == 'Backspace':
            self.command = self.command[:-1]
        else:
            self.command += event.text or ''
        self.update_prompt()

    def update_prompt(self, cursor=True):
        line = "> " + self.command
        if cursor:
            line += '_'
        if not self.console_line_started:
            self.log.write("\n" + line)
            self.console_line_started = True
        else:
            self.log.set_last_line(line)

    def clear_prompt(self):
        self.command = ""
        if self.console_line_started:
            self.log.remove_last_line()
            self.log.remove_last_line()
            self.console_line_started = False

    def run_command(self):
        cmd = self.command
        self.clear_prompt()
        self.command_history.append(cmd)
        self.interpreter(cmd)
        if self.active:
            self.update_prompt()


class GamepadThread(object):
    """Polls a gamepad on a daemon thread; each state change is dispatched as
    a GamepadEvent. Raises if no device is present (see start_gamepad)."""

    def __init__(self, dispatcher, device=None):
        if device is None:
            gamepads = inputs.devices.gamepads
            if len(gamepads) == 0:
                raise Exception("No gamepads found.")
            device = gamepads[0]
        self.dev = device
        self.dispatcher = dispatcher
        self.thread = threading.Thread(target=self._run, name='gamepad-input',
                                       daemon=True)
        self.thread.start()

    def _run(self):
        state = {}
        while True:
            for ev in self.dev.read():
                if ev.ev_type not in ('Absolute', 'Key'):
                    continue
                state[ev.code] = ev.state
                self.dispatcher.dispatch(GamepadEvent(state.copy()))


def start_gamepad(dispatcher):
    """Start gamepad polling if a device is present; never fatal.

    Returns the GamepadThread, or None when no gamepad is attached (a note is
    printed) or startup fails (the traceback is surfaced via sys.excepthook).
    """
    try:
        return GamepadThread(dispatcher)
    except Exception as exc:
        if 'No gamepads found' in str(exc):
            print("No gamepads found.")
        else:
            sys.excepthook(*sys.exc_info())
        return None
