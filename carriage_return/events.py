"""Minimal event system for game-side code.

Implements the subset of the vispy.util.event API that the game model uses,
so that game modules do not depend on any rendering library. Semantics match
vispy where the game relies on them: kwargs passed when emitting become
attributes of the event, callbacks connected most recently are called first,
duplicate connections are ignored, and delivery stops if a callback sets
event.blocked.

One deliberate difference from vispy: exceptions raised by callbacks
propagate to the emitter's caller instead of being logged and swallowed.
"""
import threading


class Event(object):
    """A single occurrence that callbacks can react to.

    All extra keyword arguments become attributes of the event.
    """
    def __init__(self, type, native=None, **kwargs):
        self._sources = []
        self._handled = False
        self._blocked = False
        self._type = type
        self._native = native
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def source(self):
        return self._sources[-1] if self._sources else None

    @property
    def type(self):
        return self._type

    @property
    def native(self):
        return self._native

    @property
    def handled(self):
        return self._handled

    @handled.setter
    def handled(self, val):
        self._handled = bool(val)

    @property
    def blocked(self):
        return self._blocked

    @blocked.setter
    def blocked(self, val):
        self._blocked = bool(val)

    def __repr__(self):
        return "<%s type=%s>" % (self.__class__.__name__, self._type)


class EventEmitter(object):
    """Emits events to connected callbacks.

    Calling the emitter with keyword arguments creates an ``event_class``
    instance carrying those kwargs as attributes and delivers it to each
    connected callback (most recently connected first).
    """
    def __init__(self, source=None, type=None, event_class=Event):
        self._callbacks = []
        self.source = source
        self.default_type = type
        self.event_class = event_class

    def connect(self, callback):
        """Add a callback; it will be invoked before previously connected ones.

        Connecting an already-connected callback is a no-op.
        """
        if callback in self._callbacks:
            return callback
        self._callbacks.insert(0, callback)
        return callback

    def disconnect(self, callback=None):
        """Remove a callback, or all callbacks if none is given."""
        if callback is None:
            self._callbacks = []
        else:
            self._callbacks.remove(callback)

    def __call__(self, *args, **kwargs):
        """Emit an event to all connected callbacks.

        Accepts either a ready-made Event instance as the only positional
        argument, or keyword arguments used to construct one.
        """
        if len(args) == 1 and isinstance(args[0], Event) and not kwargs:
            event = args[0]
        elif args:
            raise TypeError("EventEmitter accepts a single Event instance or kwargs, got %r" % (args,))
        else:
            event = self.event_class(type=self.default_type, **kwargs)

        event._sources.append(self.source)
        try:
            for callback in list(self._callbacks):
                if event.blocked:
                    break
                callback(event)
        finally:
            event._sources.pop()
        return event


class OneShotEvent(object):
    """Fires exactly once with a value; guarantees every callback runs.

    ``connect(cb)`` registers ``cb(value)`` to run when the event fires. If
    the event has already fired, ``cb`` runs immediately, on the calling
    thread, instead of being registered. ``fire(value)`` invokes every
    registered callback with ``value``, in registration order, on the firing
    thread; firing an already-fired event is an error.

    An internal lock makes the register-vs-fire race impossible: whichever
    of ``connect`` and ``fire`` gets the lock first decides whether a given
    callback runs from ``connect`` (already fired) or from ``fire``
    (registered in time) — never both, never neither. No lock is exposed to
    callers.

    Like ``EventEmitter``, an exception raised by a callback propagates to
    the caller of ``fire()`` (or ``connect()``, for the already-fired case)
    instead of being logged and swallowed; callbacks after the failing one
    do not run.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._callbacks = []
        self._value = None

    @property
    def fired(self):
        """True once the event has fired."""
        return self._event.is_set()

    def is_set(self):
        """True once the event has fired (threading.Event-style call)."""
        return self._event.is_set()

    def connect(self, cb):
        """Register ``cb(value)`` to run when the event fires.

        Runs ``cb`` immediately, on the calling thread, if the event has
        already fired.
        """
        with self._lock:
            if not self._event.is_set():
                self._callbacks.append(cb)
                return
        cb(self._value)

    def fire(self, value):
        """Invoke every registered callback with ``value``, on this thread.

        May be called exactly once; firing an already-fired event raises
        AssertionError.
        """
        with self._lock:
            assert not self._event.is_set(), "OneShotEvent already fired"
            self._value = value
            callbacks, self._callbacks = self._callbacks, []
            self._event.set()
        for cb in callbacks:
            cb(value)

    def wait(self, timeout=None):
        """Block the calling thread until fired, then return the value.

        Raises TimeoutError if *timeout* seconds elapse first, so a timeout
        can never be mistaken for a real (possibly None) fired value.
        """
        if not self._event.wait(timeout=timeout):
            raise TimeoutError("OneShotEvent.wait timed out")
        return self._value
