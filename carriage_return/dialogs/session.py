"""Threaded dialog runtime: sequential dialog loops fed by an event queue.

A modal dialog runs as a plain sequential function on its own daemon thread,
keeping its state in local variables instead of splitting logic across event
callbacks. A DialogSession *is* an input handler (input.QueuedInputHandler):
activating it on the dispatcher stack captures all user input into its queue,
and the dialog body consumes that queue:

    menu = Menu("Take which items?", items, multi_select=True)
    session = DialogSession(lambda s: run_menu(s, menu))
    session.finished.connect(lambda s: take(s.result))
    session.activate()                       # capture input (dispatcher stack)
    session.start()
    ...
    session.post(KeyPress('Down'))           # or via dispatcher.dispatch()
    session.close()                          # or force it shut

Concurrency contract
--------------------
While a modal dialog session is active, the dialog thread is the *sole
mutator* of game state; the GUI thread only reads (rendering, version-gated
sync). Input reaches the dialog only through the queue — input sources post
events and never call widget methods themselves. Observers on widgets/layers
may therefore be invoked from the dialog thread and must be cheap and
thread-safe (game-side painters repaint numpy grids; backend observers just
set a dirty flag).

Completion and errors
---------------------
When the dialog body returns, ``session.result`` holds its return value and
``session.finished`` — an ``events.OneShotEvent`` — fires with the session
itself, **from the dialog thread**; every callback registered with
``session.finished.connect()`` is then called with the session, in
registration order. Callbacks connected after completion run immediately, on
the connecting thread (see ``OneShotEvent``).

``session.get()`` raises DialogClosed when a Close event arrives, so every
dialog loop unwinds correctly without each author remembering to check for
it; the runner catches DialogClosed (result stays None). Any *other*
exception from the dialog body is recorded as ``session.error``, completion
bookkeeping still runs (so the UI can drop the dialog's grid; result is
None), and the exception is then re-raised in the dialog thread so it
surfaces loudly via threading.excepthook rather than being silently
swallowed.
"""
import threading

from ..events import OneShotEvent
from ..input import Close, QueuedInputHandler


class DialogClosed(Exception):
    """Raised by DialogSession.get() when the session is closed externally."""


class DialogSession(QueuedInputHandler):
    """Runs a dialog body ``fn(session)`` on its own daemon thread.

    The body pulls InputEvents with ``session.get()`` and mutates its widgets
    in a plain loop; its return value becomes ``session.result``. As a
    QueuedInputHandler, the session sits directly on the dispatcher stack
    while active and consumes every event into its queue. See the module
    docstring for the threading contract.
    """

    def __init__(self, fn, name=None):
        QueuedInputHandler.__init__(self)
        self.fn = fn
        self.name = name or getattr(fn, '__name__', 'dialog')
        self.result = None
        self.error = None
        self.finished = OneShotEvent()
        self.thread = None

    def start(self):
        """Spawn the dialog thread. May be called once."""
        assert self.thread is None, "DialogSession already started"
        self.thread = threading.Thread(target=self._run, name="dialog-%s" % self.name,
                                       daemon=True)
        self.thread.start()
        return self

    def post(self, event):
        """Enqueue an InputEvent for the dialog loop; never blocks.

        Callable from any thread. Events posted after completion are ignored
        by virtue of nothing consuming them.
        """
        self.queue.put(event)

    def close(self):
        """Ask the dialog loop to unwind (idempotent, any thread)."""
        self.post(Close())

    def get(self, timeout=None):
        """Block the dialog thread until the next InputEvent arrives.

        Raises DialogClosed when a Close event arrives, so dialog loops
        unwind without checking for it explicitly. Raises queue.Empty on
        timeout, exactly like Queue.get.
        """
        event = self.queue.get(timeout=timeout)
        if isinstance(event, Close):
            raise DialogClosed()
        return event

    def join(self, timeout=None):
        """Wait for the dialog thread to exit (mainly for tests)."""
        self.thread.join(timeout=timeout)

    def _run(self):
        try:
            self.result = self.fn(self)
        except DialogClosed:
            pass  # result stays None
        except BaseException as exc:
            self.error = exc
            self._finish()
            raise  # surfaces via threading.excepthook; never swallowed
        self._finish()

    def _finish(self):
        self.finished.fire(self)
