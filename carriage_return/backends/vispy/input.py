"""Vispy canvas keyboard events -> normalized InputEvents -> dispatcher.

This is the keyboard input source for the vispy backend: the *only* place
native vispy key events are seen. Native key objects are normalized to plain
string names here, at the boundary, so everything game-side deals in strings
(vispy Key objects compare equal to their name string but are not reliably
hashable across versions).

Keyboard auto-repeat is also dropped here. X11/Qt delivers a held key as a
stream of release/press pairs, which game-side would read as the key being
let go and pressed again many times a second — movement would keep restarting
from a standstill. Filtering at the boundary lets the rest of the game treat a
key as simply down or up.
"""
from ...input import KeyPress, KeyRelease


def _key_name(key):
    """Normalize a native key object to its plain string name.

    vispy names letter keys with their uppercase character ('T'), but the
    game side compares against lowercase physical-key names ('t'); vispy Key
    objects hid this by comparing case-insensitively, plain strings do not.
    Lowercase single-character names so 't'/'r'/'d' etc. reach their handlers,
    while multi-character names ('Right', 'Escape', 'Shift') pass through.
    """
    name = getattr(key, 'name', key)
    if isinstance(name, str) and len(name) == 1:
        return name.lower()
    return name


def _is_auto_repeat(event):
    """True if *event* is a synthetic auto-repeat rather than a real press.

    Backends that do not expose the flag report False, i.e. every event is
    taken at face value.
    """
    native = getattr(event, 'native', None)
    is_auto_repeat = getattr(native, 'isAutoRepeat', None)
    return bool(is_auto_repeat()) if is_auto_repeat is not None else False


class CanvasInputSource(object):
    """Connects a vispy SceneCanvas's key events to an InputDispatcher.

    Runs on the GUI thread; per the input contract it only constructs events
    and dispatches them — handlers enqueue, nothing mutates game state here.
    """

    def __init__(self, canvas, dispatcher):
        self.canvas = canvas
        self.dispatcher = dispatcher
        canvas.events.key_press.connect(self._key_pressed)
        canvas.events.key_release.connect(self._key_released)

    def _key_pressed(self, event):
        if _is_auto_repeat(event):
            return
        self.dispatcher.dispatch(KeyPress(_key_name(event.key),
                                          getattr(event, 'text', None)))

    def _key_released(self, event):
        if _is_auto_repeat(event):
            return
        self.dispatcher.dispatch(KeyRelease(_key_name(event.key),
                                            getattr(event, 'text', None)))
