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
