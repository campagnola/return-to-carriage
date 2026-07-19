"""The vispy keyboard boundary: key-name normalization and auto-repeat.

Only the pure event-translation helpers are exercised here — no canvas, no
GL context. Held keys arrive from X11/Qt as streams of release/press pairs;
dropping them here is what lets the game treat a key as simply down or up.
"""
from carriage_return.backends.vispy.input import _key_name, _is_auto_repeat


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
