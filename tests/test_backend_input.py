"""The vispy input boundary: key-name normalization, auto-repeat, and mouse
events.

Only the pure event-translation helpers are exercised here (mouse events via
a fake canvas) — no GL context. Held keys arrive from X11/Qt as streams of release/press pairs;
dropping them here is what lets the game treat a key as simply down or up.
"""
from carriage_return.backends.vispy.input import CanvasInputSource, _key_name, _is_auto_repeat
from carriage_return.input import MouseClick, MouseWheel


class FakeKey(object):
    def __init__(self, name):
        self.name = name


class FakeNative(object):
    def __init__(self, auto_repeat):
        self._auto_repeat = auto_repeat

    def isAutoRepeat(self):
        return self._auto_repeat


class FakeEvent(object):
    def __init__(self, native=None):
        self.key = FakeKey('Right')
        self.native = native


def test_key_name_normalizes_native_key_objects():
    assert _key_name(FakeKey('Right')) == 'Right'
    assert _key_name('Right') == 'Right'  # already a plain string


def test_auto_repeat_is_detected():
    assert _is_auto_repeat(FakeEvent(FakeNative(True))) is True
    assert _is_auto_repeat(FakeEvent(FakeNative(False))) is False


def test_events_without_the_flag_are_taken_at_face_value():
    assert _is_auto_repeat(FakeEvent(native=None)) is False
    assert _is_auto_repeat(FakeEvent(native=object())) is False


class FakeEmitter(object):
    def __init__(self):
        self.callbacks = []

    def connect(self, callback, position='last'):
        self.callbacks.append((callback, position))


class FakeCanvas(object):
    def __init__(self):
        self.events = type('Events', (), {
            name: FakeEmitter()
            for name in ('key_press', 'key_release', 'mouse_wheel', 'mouse_press')})()


class FakeDispatcher(object):
    def __init__(self, consumer):
        self.consumer = consumer
        self.dispatched = []

    def dispatch(self, event):
        self.dispatched.append(event)
        return self.consumer


class FakeMouseEvent(object):
    def __init__(self, pos=(30, 50), delta=(0, 1), button=1):
        self.pos = pos
        self.delta = delta
        self.button = button
        self.blocked = None


def make_source(consumer):
    canvas, dispatcher = FakeCanvas(), FakeDispatcher(consumer)
    source = CanvasInputSource(canvas, dispatcher, lambda pos: ('grid', pos[1], pos[0]))
    return canvas, dispatcher, source


def test_mouse_handlers_run_before_the_scene_sees_the_event():
    canvas, _, _ = make_source(None)
    for emitter in (canvas.events.mouse_wheel, canvas.events.mouse_press):
        assert [pos for _, pos in emitter.callbacks] == ['first']


def test_wheel_is_dispatched_at_its_cell_and_blocked_only_if_consumed():
    _, dispatcher, source = make_source(consumer=object())
    event = FakeMouseEvent(pos=(30, 50), delta=(0, -2))
    source._mouse_wheel(event)
    (sent,) = dispatcher.dispatched
    assert isinstance(sent, MouseWheel)
    assert (sent.grid, sent.row, sent.col, sent.steps) == ('grid', 50, 30, -2)
    assert event.blocked is True

    _, _, source = make_source(consumer=None)
    event = FakeMouseEvent()
    source._mouse_wheel(event)
    assert event.blocked is False


def test_press_button_names_are_normalized():
    _, dispatcher, source = make_source(consumer=None)
    for button in (1, 2, 3):
        source._mouse_pressed(FakeMouseEvent(button=button))
    assert all(isinstance(ev, MouseClick) for ev in dispatcher.dispatched)
    assert [ev.button for ev in dispatcher.dispatched] == ['left', 'right', 'middle']
