"""Dispatcher stack, focus events, and the threaded gameplay/command handlers.

Everything here is headless and (except two thread smoke tests) synchronous:
GameplayInputHandler and CommandInputHandler are built with
``start_thread=False`` and driven through their ``_process`` methods with an
injected fake clock, so movement-repeat timing is tested without sleeping.
"""
import queue

import pytest

from carriage_return.input import (
    InputDispatcher, InputHandler, QueuedInputHandler, GameplayInputHandler,
    CommandInputHandler, KeyPress, KeyRelease, GamepadEvent, FocusIn, FocusOut,
    Close)
from carriage_return.scene import MessageLog


class RecordingHandler(InputHandler):
    def __init__(self, consume=True):
        self.consume = consume
        self.events = []

    def handle(self, ev):
        self.events.append(ev)
        return self.consume

    def of_type(self, cls):
        return [ev for ev in self.events if isinstance(ev, cls)]

    def pressed(self):
        return [ev.key for ev in self.of_type(KeyPress)]


@pytest.fixture
def dispatcher():
    InputDispatcher.reset()
    disp = InputDispatcher()
    yield disp
    InputDispatcher.reset()


def test_singleton(dispatcher):
    with pytest.raises(Exception):
        InputDispatcher()
    InputDispatcher.reset()
    disp2 = InputDispatcher()
    assert InputDispatcher.dispatcher is disp2


def test_stack_dispatch_order_and_consumption(dispatcher):
    bottom = RecordingHandler(consume=True)
    top = RecordingHandler(consume=False)
    dispatcher.add_handler(bottom)
    dispatcher.add_handler(top)

    # top sees the event first; since it does not consume, bottom sees it too
    consumed_by = dispatcher.dispatch(KeyPress('a'))
    assert top.pressed() == ['a']
    assert bottom.pressed() == ['a']
    assert consumed_by is bottom

    # a consuming top handler stops propagation
    top.consume = True
    assert dispatcher.dispatch(KeyPress('b')) is top
    assert top.pressed() == ['a', 'b']
    assert bottom.pressed() == ['a']


def test_unconsumed_event_returns_none(dispatcher):
    h = RecordingHandler(consume=False)
    dispatcher.add_handler(h)
    assert dispatcher.dispatch(KeyPress('a')) is None


def test_focus_out_on_add_focus_in_on_remove(dispatcher):
    a = RecordingHandler()
    b = RecordingHandler()
    dispatcher.add_handler(a)
    assert a.events == []  # nothing displaced

    dispatcher.add_handler(b)
    assert len(a.of_type(FocusOut)) == 1  # a lost the top
    assert b.events == []

    dispatcher.remove_handler(b)
    assert len(a.of_type(FocusIn)) == 1  # a regained the top


def test_no_focus_events_when_top_unchanged(dispatcher):
    a = RecordingHandler()
    b = RecordingHandler()
    dispatcher.add_handler(a)
    dispatcher.add_handler(b)
    a.events.clear()
    b.events.clear()

    # removing a non-top handler leaves the top alone: no focus events
    dispatcher.remove_handler(a)
    assert a.events == [] and b.events == []


def test_queued_handler_captures_everything(dispatcher):
    q = QueuedInputHandler()
    below = RecordingHandler()
    dispatcher.add_handler(below)
    dispatcher.add_handler(q)

    for ev in [KeyPress('a'), KeyRelease('a'), GamepadEvent({}), Close()]:
        assert dispatcher.dispatch(ev) is q
    assert below.pressed() == []
    drained = []
    while True:
        try:
            drained.append(q.queue.get_nowait())
        except queue.Empty:
            break
    assert [type(ev) for ev in drained] == [KeyPress, KeyRelease, GamepadEvent, Close]


def test_activate_deactivate(dispatcher):
    h = RecordingHandler()
    assert not h.active
    h.activate()
    assert h.active
    h.activate()  # no duplicate
    assert dispatcher.handlers.count(h) == 1
    h.deactivate()
    assert not h.active


# -- GameplayInputHandler ----------------------------------------------------

class FakeClock(object):
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


class StubScene(object):
    def __init__(self):
        self.quit_requested = False


class RecordingInterpreter(object):
    """Records the take/read/drop calls the t/r/d shortcuts make."""
    def __init__(self):
        self.calls = []

    def take(self, args):
        self.calls.append(('take', args))

    def read(self, args):
        self.calls.append(('read', args))

    def drop(self, args):
        self.calls.append(('drop', args))


class StubDM(object):
    """Accepts every move, and — as the real DM does — actually relocates the
    player, so the handler's float position tracks a moving target."""
    def __init__(self):
        self.scene = StubScene()
        self.moves = []

    def request_player_move(self, player, pos):
        self.moves.append(tuple(pos))
        player.location.slot = tuple(pos)


class StubPlayer(object):
    class Location(object):
        def __init__(self):
            self.slot = (5, 5)

    def __init__(self):
        self.location = StubPlayer.Location()


def make_gameplay(dispatcher, **kwds):
    clock = kwds.pop('clock', FakeClock())
    dm = StubDM()
    handler = GameplayInputHandler(dm, StubPlayer(), clock=clock,
                                   start_thread=False, **kwds)
    dispatcher.add_handler(handler)
    return handler, dm, clock


def _drain(handler, clock):
    """Feed every velocity already sitting in ``movement_queue`` to the pacer
    at the current fake time, then process any ``MovementTick`` that
    produces -- the instantaneous part of what the real pacer/gameplay
    threads do before either ever blocks on a timeout."""
    while True:
        try:
            velocity = handler.movement_queue.get_nowait()
        except queue.Empty:
            break
        handler.pacer.step(velocity, clock.t)
    while True:
        try:
            handler._process(handler.queue.get_nowait())
        except queue.Empty:
            break


def tick(handler, clock, duration, dt=0.005):
    """Run the movement pacer, as its own thread would, for *duration*
    seconds of fake time, delivering queued velocities and any
    MovementTicks they produce along the way."""
    _drain(handler, clock)
    elapsed = 0.0
    while elapsed < duration:
        clock.t += dt
        elapsed += dt
        handler.pacer.step(None, clock.t)
        _drain(handler, clock)


def test_first_step_waits_for_the_start_delay(dispatcher):
    handler, dm, clock = make_gameplay(dispatcher)

    # the press alone does not move: the grace period is there so the second
    # key of a diagonal can still arrive
    handler._process(KeyPress('Right'))
    assert dm.moves == []

    tick(handler, clock, handler.start_delay + 0.01)
    assert dm.moves == [(6, 5)]


def test_walk_tap_gives_exactly_one_step(dispatcher):
    handler, dm, clock = make_gameplay(dispatcher)
    handler._process(KeyPress('Right'))
    tick(handler, clock, 0.12)  # a tap; the second step is not due until ~0.2s
    handler._process(KeyRelease('Right'))
    assert dm.moves == [(6, 5)]
    assert handler._movement_direction() == (0, 0)


def test_holding_an_arrow_moves_at_walking_speed(dispatcher):
    """The bug this model replaced: an orthogonal hold stopped after a step."""
    handler, dm, clock = make_gameplay(dispatcher)
    handler._process(KeyPress('Right'))
    tick(handler, clock, 1.0)

    # one immediate step, then walk_speed (6 cells/s) of steady travel
    assert 6 <= len(dm.moves) <= 8
    xs = [move[0] for move in dm.moves]
    assert xs == sorted(xs) and len(set(xs)) == len(xs)  # one cell at a time
    assert all(move[1] == 5 for move in dm.moves)  # no drift off the axis


def test_running_is_faster_than_walking(dispatcher):
    walker, walk_dm, walk_clock = make_gameplay(dispatcher)
    walker._process(KeyPress('Right'))
    tick(walker, walk_clock, 1.0)

    runner, run_dm, run_clock = make_gameplay(dispatcher)
    runner._process(KeyPress('Shift'))
    runner._process(KeyPress('Right'))
    tick(runner, run_clock, 1.0)

    assert len(run_dm.moves) > len(walk_dm.moves)
    assert 16 <= len(run_dm.moves) <= 20  # run_speed is 18 cells/s


def test_shift_mid_hold_switches_to_running(dispatcher):
    handler, dm, clock = make_gameplay(dispatcher)
    handler._process(KeyPress('Right'))
    tick(handler, clock, 1.0)
    walked = len(dm.moves)

    handler._process(KeyPress('Shift'))
    tick(handler, clock, 1.0)
    assert len(dm.moves) - walked > walked


def test_opposing_arrows_cancel(dispatcher):
    handler, dm, clock = make_gameplay(dispatcher)
    handler._process(KeyPress('Right'))
    handler._process(KeyPress('Left'))
    tick(handler, clock, 1.0)
    assert dm.moves == []
    assert handler.pacer.hold_start is None  # treated as stopped


def test_hold_state_is_dropped_when_stopped(dispatcher):
    """A partial hold must not be banked across a pause."""
    handler, dm, clock = make_gameplay(dispatcher)
    handler._process(KeyPress('Right'))
    tick(handler, clock, 0.1)  # first step, plus a fraction of the next
    handler._process(KeyRelease('Right'))
    _drain(handler, clock)
    assert handler.pacer.hold_start is None

    moved = len(dm.moves)
    handler._process(KeyPress('Right'))
    tick(handler, clock, handler.start_delay + 0.005)
    # exactly one step: the new hold starts from a clean cell
    assert len(dm.moves) == moved + 1


def test_diagonal_starts_diagonal(dispatcher):
    handler, dm, clock = make_gameplay(dispatcher)
    handler._process(KeyPress('Right'))
    clock.t += 0.01  # the second arrow lands a moment later, as fingers do
    handler._process(KeyPress('Up'))
    tick(handler, clock, 0.05)

    # no stray orthogonal step: every step so far is diagonal
    assert dm.moves == [(6, 6)]


def test_release_of_one_arrow_leaves_orthogonal_movement(dispatcher):
    handler, dm, clock = make_gameplay(dispatcher)
    handler._process(KeyPress('Right'))
    handler._process(KeyPress('Up'))
    tick(handler, clock, 1.0)
    x, y = dm.moves[-1]
    assert x - 5 == y - 5 > 0  # travelled evenly along both axes

    handler._process(KeyRelease('Up'))
    tick(handler, clock, 0.3)
    assert dm.moves[-1][0] > x       # still moving in x
    assert dm.moves[-1][1] == y      # and no longer in y


class NarrowGapDM(StubDM):
    """A gap passable only by the exact diagonal move (6, 7) from (5, 6):
    both cardinal neighbors are walled off, mirroring the real DM's
    diagonal-only ``walkable`` check for a narrow hall."""
    BLOCKED = {(5, 7), (6, 6)}

    def request_player_move(self, player, pos):
        pos = tuple(int(v) for v in pos)
        if pos in self.BLOCKED:
            return
        self.moves.append(pos)
        player.location.slot = pos


def test_diagonal_added_mid_hold_passes_a_narrow_gap(dispatcher):
    """The reported bug: holding Up alone (already moving), then adding a
    side key, must combine into one atomic diagonal step immediately -- not
    drift each axis independently -- or a gap passable only by a true
    diagonal move is never reached."""
    handler, dm, clock = make_gameplay(dispatcher)
    handler.dm = NarrowGapDM()

    handler._process(KeyPress('Up'))
    tick(handler, clock, 0.3)  # already moving, and walled off at (5, 7)
    assert handler.player.location.slot == (5, 6)

    handler._process(KeyPress('Right'))
    _drain(handler, clock)  # combined direction takes effect immediately
    assert handler.player.location.slot == (6, 7)  # through the diagonal-only gap


class SlidingDM(StubDM):
    """A DM with a wall along x: the x component of a move is always refused."""
    def request_player_move(self, player, pos):
        self.moves.append(tuple(pos))
        player.location.slot = (player.location.slot[0], pos[1])


def test_wall_slides_instead_of_stalling(dispatcher):
    handler, dm, clock = make_gameplay(dispatcher)
    handler.dm = SlidingDM()

    handler._process(KeyPress('Right'))
    handler._process(KeyPress('Up'))
    tick(handler, clock, 1.0)

    # x is walled off, so the player only ever advances in y — and keeps
    # advancing: the float position follows the real one instead of burrowing
    assert handler.player.location.slot[0] == 5
    assert handler.player.location.slot[1] > 8


def test_gamepad_movement(dispatcher):
    handler, dm, clock = make_gameplay(dispatcher)
    handler._process(GamepadEvent({'ABS_HAT0X': 1}))
    tick(handler, clock, 0.1)
    assert dm.moves[-1] == (6, 5)

    handler._process(GamepadEvent({'ABS_HAT0X': 0}))
    assert handler._movement_direction() == (0, 0)


def test_focus_out_clears_held_state(dispatcher):
    handler, dm, clock = make_gameplay(dispatcher)
    handler._process(KeyPress('Right'))
    handler._process(GamepadEvent({'ABS_HAT0X': 1}))
    tick(handler, clock, 0.1)
    moved = len(dm.moves)

    handler._process(FocusOut())
    assert handler.keys == set() and handler.gamepad_state == {}
    tick(handler, clock, 1.0)
    assert len(dm.moves) == moved  # nothing held, nothing moves


def test_no_events_reach_gameplay_under_dialog(dispatcher):
    handler, dm, clock = make_gameplay(dispatcher)
    dialog = QueuedInputHandler()
    dispatcher.add_handler(dialog)

    # the push delivered FocusOut to the gameplay handler...
    assert isinstance(handler.queue.get_nowait(), FocusOut)

    # ...and while the dialog is on top, events never reach gameplay
    dispatcher.dispatch(KeyPress('Right'))
    dispatcher.dispatch(KeyPress('t'))
    with pytest.raises(queue.Empty):
        handler.queue.get_nowait()

    dispatcher.remove_handler(dialog)
    assert isinstance(handler.queue.get_nowait(), FocusIn)


def test_action_keys(dispatcher):
    interp = RecordingInterpreter()
    handler, dm, clock = make_gameplay(dispatcher, interpreter=interp)
    handler._process(KeyPress('t', 't'))
    handler._process(KeyPress('r', 'r'))
    handler._process(KeyPress('d', 'd'))
    # t/r/d all route through the shared action layer with no args
    assert interp.calls == [('take', []), ('read', []), ('drop', [])]


def test_escape_requests_quit(dispatcher):
    handler, dm, clock = make_gameplay(dispatcher)
    handler._process(KeyPress('Escape'))
    assert dm.scene.quit_requested is True


def test_tab_toggles_command_handler(dispatcher):
    log = MessageLog()
    cmd = CommandInputHandler(log, lambda c: None, start_thread=False)
    handler, dm, clock = make_gameplay(dispatcher, command_handler=cmd)

    handler._process(KeyPress('Tab'))
    assert cmd.active
    assert log.lines[-1] == "> _"  # prompt drawn on activation

    handler._process(KeyPress('Tab'))
    assert not cmd.active
    assert all(not line.startswith("> ") for line in log.lines)


def test_gameplay_thread_drains_queue(dispatcher):
    """Smoke test of the real thread: Escape via dispatch() must take effect."""
    dm = StubDM()
    handler = GameplayInputHandler(dm, StubPlayer())  # real thread
    dispatcher.add_handler(handler)
    dispatcher.dispatch(KeyPress('Escape'))

    import time
    deadline = time.time() + 5.0
    while not dm.scene.quit_requested and time.time() < deadline:
        time.sleep(0.005)
    assert dm.scene.quit_requested is True


# -- CommandInputHandler -----------------------------------------------------

def make_command(dispatcher):
    log = MessageLog()
    commands = []
    handler = CommandInputHandler(log, commands.append, start_thread=False)
    return handler, log, commands


def test_command_prompt_editing(dispatcher):
    handler, log, commands = make_command(dispatcher)
    handler.activate()
    assert log.lines[-1] == "> _"

    for char in 'go':
        handler._process(KeyPress(char, char))
    assert log.lines[-1] == "> go_"

    handler._process(KeyPress('Backspace'))
    assert log.lines[-1] == "> g_"

    handler._process(KeyPress('Enter'))
    assert commands == ['g']
    assert log.lines[-1] == "> _"  # still active: fresh prompt
    assert handler.command_history == ['g']

    handler.deactivate()
    assert all(not line.startswith("> ") for line in log.lines)


def test_command_handler_refuses_escape_and_tab(dispatcher):
    handler, log, commands = make_command(dispatcher)
    below = RecordingHandler()
    dispatcher.add_handler(below)
    handler.activate()

    # Escape/Tab fall through to the handler below; other keys are consumed
    assert dispatcher.dispatch(KeyPress('Tab')) is below
    assert dispatcher.dispatch(KeyPress('Escape')) is below
    assert dispatcher.dispatch(KeyPress('x', 'x')) is handler
    assert below.pressed() == ['Tab', 'Escape']


def test_command_thread_runs_interpreter(dispatcher):
    """Smoke test of the real thread: typed command reaches the interpreter."""
    log = MessageLog()
    commands = []
    handler = CommandInputHandler(log, commands.append)  # real thread
    handler.activate()
    for char in 'hi':
        handler.handle(KeyPress(char, char))
    handler.handle(KeyPress('Enter'))

    import time
    deadline = time.time() + 5.0
    while not commands and time.time() < deadline:
        time.sleep(0.005)
    assert commands == ['hi']
