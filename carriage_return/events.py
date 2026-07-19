"""Minimal event system for game-side code, derived from vispy's event system.
"""
import threading


class Event(object):
    """A single occurrence that callbacks can react to.

    All extra keyword arguments become attributes of the event.
    """
    def __init__(self, type, native=None, source=None, **kwargs):
        self._source = source
        self._handled = False
        self._type = type
        self._native = native
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def source(self):
        """The object that emitted this event, if any."""
        return self._source

    @property
    def type(self):
        return self._type

    @property
    def native(self):
        """The native event that triggered this event, if any."""
        return self._native

    @property
    def handled(self):
        """True if a callback has handled this event and no further processing is needed.
        
        Note: handled events are still propagated to all listeners; it is the listener's job
        to check the handled flag and decide whether to act on the event.
        """
        return self._handled

    @handled.setter
    def handled(self, val):
        self._handled = bool(val)

    def __repr__(self):
        return "<%s type=%s>" % (self.__class__.__name__, self._type)
    

class Observable:
    """An object that invokes callbacks when events occur::

        some_event = Observable(source=self)
        some_event.connect(callback)  # callback will be invoked when the event fires
        some_event(value=42)  # invokes all callbacks with **{'value': 42, 'source': self}
                    
    """
    def __init__(self, **kwargs):
        self.callbacks = []
        self.kwargs = kwargs
        self.blocked = False

    def connect(self, callback):
        """Add a callback; it will be invoked before previously connected ones.

        Connecting an already-connected callback is a no-op.
        """
        if callback in self.callbacks:
            return callback
        self.callbacks.insert(0, callback)
        return callback

    def disconnect(self, callback=None):
        """Remove a callback, or all callbacks if none is given."""
        if callback is None:
            self.callbacks = []
        else:
            self.callbacks.remove(callback)

    def __call__(self, *args, **kwargs):
        # note: __call__ may be invoked very frequently, so we avoid all unnecessary work here.

        if self.blocked or len(self.callbacks) == 0:
            return

        for callback in list(self.callbacks):
            callback(*args, **kwargs, **self.kwargs)


class EventEmitter(Observable):
    """Observable that emits Event instances::

        event = EventEmitter(source=self, type='my_event', event_class=MyEvent)
        event.connect(callback)  # callback will be invoked when the event fires
        event(value=42)  # emits MyEvent(value=42, source=self, type='my_event') to all callbacks
    
    """
    def __init__(self, event_class=Event, **kwargs):
        super().__init__(**kwargs)
        self.event_class = event_class

    def __call__(self, *args, **kwargs):
        """Emit an event to all connected callbacks.
        """
        if self.blocked or len(self.callbacks) == 0:
            return

        event = self.event_class(*args, **kwargs, **self.kwargs)

        for callback in list(self.callbacks):
            callback(event)

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
