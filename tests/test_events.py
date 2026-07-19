import threading

import pytest

from carriage_return.events import Event, EventEmitter, Observable, OneShotEvent


class MyEvent(Event):
    pass


def test_kwargs_become_attributes():
    ev = Event(type='thing_happened', foo=1, bar='x')
    assert ev.type == 'thing_happened'
    assert ev.foo == 1
    assert ev.bar == 'x'


def test_emit_creates_event_with_class_type_and_source():
    src = object()
    em = EventEmitter(source=src, type='location_change', event_class=MyEvent)
    received = []
    em.connect(received.append)

    em(old_location=(None, None))

    assert len(received) == 1
    ev = received[0]
    assert isinstance(ev, MyEvent)
    assert ev.type == 'location_change'
    assert ev.old_location == (None, None)
    # the emitter's kwargs are baked into every event it emits
    assert ev.source is src


def test_connect_disconnect():
    em = EventEmitter(type='t')
    calls = []

    def cb(ev):
        calls.append(ev)

    em.connect(cb)
    em()
    assert len(calls) == 1

    em.disconnect(cb)
    em()
    assert len(calls) == 1

    with pytest.raises(ValueError):
        em.disconnect(cb)


def test_duplicate_connect_ignored():
    em = EventEmitter(type='t')
    calls = []

    def cb(ev):
        calls.append(ev)

    em.connect(cb)
    em.connect(cb)
    em()
    assert len(calls) == 1


def test_most_recently_connected_called_first():
    em = EventEmitter(type='t')
    order = []
    em.connect(lambda ev: order.append('a'))
    em.connect(lambda ev: order.append('b'))
    em()
    assert order == ['b', 'a']


def test_handled_does_not_stop_delivery():
    """``handled`` is advisory: every callback still sees the event and decides
    for itself whether to act on it."""
    em = EventEmitter(type='t')
    order = []

    def handler(ev):
        order.append('handler')
        ev.handled = True

    em.connect(lambda ev: order.append(('late', ev.handled)))
    em.connect(handler)  # called first
    em()
    assert order == ['handler', ('late', True)]


def test_blocked_emitter_delivers_nothing():
    em = EventEmitter(type='t')
    calls = []
    em.connect(calls.append)

    em.blocked = True
    em()
    assert calls == []

    em.blocked = False
    em()
    assert len(calls) == 1


def test_observable_passes_kwargs_through():
    """Observable broadcasts plain kwargs -- no Event instance involved."""
    obs = Observable(source='s')
    received = []
    obs.connect(lambda **kw: received.append(kw))

    obs(value=42)

    assert received == [{'value': 42, 'source': 's'}]


def test_callback_exception_propagates():
    em = EventEmitter(type='t')

    def bad(ev):
        raise RuntimeError("boom")

    em.connect(bad)
    with pytest.raises(RuntimeError):
        em()


def test_oneshot_connect_then_fire():
    ev = OneShotEvent()
    received = []
    ev.connect(received.append)
    assert not ev.fired
    assert not ev.is_set()

    ev.fire('value')

    assert received == ['value']
    assert ev.fired
    assert ev.is_set()


def test_oneshot_connect_after_fire_runs_immediately():
    ev = OneShotEvent()
    ev.fire('value')

    received = []
    ev.connect(received.append)

    assert received == ['value']


def test_oneshot_cross_thread_wait():
    ev = OneShotEvent()
    fired_from = []

    def fire_later():
        fired_from.append(threading.current_thread())
        ev.fire('done')

    thread = threading.Thread(target=fire_later)
    thread.start()

    result = ev.wait(timeout=10.0)
    thread.join(timeout=10.0)

    assert result == 'done'
    assert fired_from == [thread]


def test_oneshot_wait_timeout_raises():
    ev = OneShotEvent()
    with pytest.raises(TimeoutError):
        ev.wait(timeout=0.01)


def test_oneshot_double_fire_is_an_error():
    ev = OneShotEvent()
    ev.fire('first')
    with pytest.raises(AssertionError):
        ev.fire('second')


def test_oneshot_callback_exception_propagates():
    ev = OneShotEvent()

    def bad(value):
        raise RuntimeError("boom")

    ev.connect(bad)
    with pytest.raises(RuntimeError):
        ev.fire('value')

    # already fired: a late connect that raises also propagates to the caller
    ev2 = OneShotEvent()
    ev2.fire('value')
    with pytest.raises(RuntimeError):
        ev2.connect(bad)
