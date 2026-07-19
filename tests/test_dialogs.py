"""Headless tests for the threaded dialog runtime (no vispy/GUI involved)."""
import threading

import pytest

from carriage_return.dialogs import (DialogSession, DialogClosed, run_menu,
                                     run_pager, Menu, MenuItem, Pager)
from carriage_return.input import Close, KeyPress, KeyRelease

JOIN_TIMEOUT = 10.0


def press(key):
    return KeyPress(key)


def finish(session):
    session.join(timeout=JOIN_TIMEOUT)
    assert session.finished.is_set()
    assert not session.thread.is_alive()


def test_run_menu_multi_select():
    menu = Menu("take", [MenuItem('a sword', value=1),
                         MenuItem('a shield', value=2),
                         MenuItem('a torch', value=3)], multi_select=True)
    session = DialogSession(lambda s: run_menu(s, menu)).start()

    for key in ['Down', 'Down', 'Space', 'Enter']:
        session.post(press(key))
    finish(session)

    assert session.result == [3]
    assert menu.done
    assert menu.cursor == 2
    assert [item.checked for item in menu.items] == [False, False, True]


def test_run_menu_single_select_and_key_release_ignored():
    menu = Menu("choose", ['x', 'y', 'z'])
    session = DialogSession(lambda s: run_menu(s, menu)).start()

    session.post(press('Down'))
    session.post(KeyRelease('Down'))  # must be ignored
    session.post(press('Enter'))
    finish(session)

    assert session.result == 'y'
    assert menu.result == 'y'


def test_run_menu_escape_cancels():
    menu = Menu("choose", ['x', 'y'])
    session = DialogSession(lambda s: run_menu(s, menu)).start()

    session.post(press('Down'))
    session.post(press('Escape'))
    finish(session)

    assert session.result is None
    assert menu.done
    assert menu.result is None


def test_close_unwinds_via_dialog_closed():
    seen = []
    menu = Menu("choose", ['x', 'y'])
    session = DialogSession(lambda s: run_menu(s, menu))
    session.finished.connect(seen.append)
    session.start()

    session.post(press('Down'))
    session.close()
    finish(session)

    assert session.result is None
    assert seen == [session]
    assert not menu.done  # unwound externally; the widget was never accepted


def test_get_raises_dialog_closed_directly():
    session = DialogSession(lambda s: None)
    session.post(Close())
    with pytest.raises(DialogClosed):
        session.get(timeout=1.0)


def test_finished_connect_before_and_after_runs_on_dialog_thread():
    threads = []

    def body(session):
        return 42

    session = DialogSession(body)
    session.finished.connect(lambda s: threads.append(threading.current_thread()))
    session.start()
    finish(session)

    assert session.result == 42
    assert threads == [session.thread]

    # connected after completion: fires immediately, on the calling thread
    late = []
    session.finished.connect(lambda s: late.append((s.result, threading.current_thread())))
    assert late == [(42, threading.current_thread())]


def test_run_pager():
    pager = Pager("book", ['page one', 'page two', 'page three'])
    session = DialogSession(lambda s: run_pager(s, pager)).start()

    for key in ['Right', 'Right', 'Right', 'Left', 'Escape']:
        session.post(press(key))
    finish(session)

    assert session.result is None
    assert pager.done
    assert pager.page == 1  # 0 ->1 ->2 ->clamped 2 ->1


def test_dialog_body_error_recorded_and_finished_still_fires():
    seen = []
    hooked = []

    def body(session):
        raise RuntimeError("boom")

    session = DialogSession(body)
    session.finished.connect(seen.append)

    orig_hook = threading.excepthook
    threading.excepthook = lambda args: hooked.append(args)
    try:
        session.start()
        finish(session)
    finally:
        threading.excepthook = orig_hook

    assert isinstance(session.error, RuntimeError)
    assert session.result is None
    assert seen == [session]
    # the exception was re-raised in the dialog thread, not swallowed
    assert len(hooked) == 1 and hooked[0].exc_type is RuntimeError
