"""Unit tests for the game-side dialog widget models (carriage_return/dialogs)."""
import pytest

from carriage_return.dialogs import Menu, MenuItem, Pager


class VersionWatcher(object):
    """Record observer calls and check they accompany version bumps."""

    def __init__(self, widget):
        self.widget = widget
        self.calls = 0
        widget.observer = self._called

    def _called(self):
        self.calls += 1


def test_menu_items_auto_wrapped():
    menu = Menu("take", ["a scroll", MenuItem("a torch", value=42)])
    assert all(isinstance(item, MenuItem) for item in menu.items)
    assert menu.items[0].label == "a scroll"
    assert menu.items[0].value == "a scroll"   # value defaults to label
    assert menu.items[1].value == 42
    assert len(menu) == 2


def test_cursor_movement_clamped():
    menu = Menu("m", ["a", "b", "c"])
    assert menu.cursor == 0
    menu.move(-1)
    assert menu.cursor == 0
    menu.move(1)
    assert menu.cursor == 1
    menu.move(10)
    assert menu.cursor == 2
    menu.move(-10)
    assert menu.cursor == 0


def test_toggle_checked_bookkeeping():
    menu = Menu("m", ["a", "b"], multi_select=True)
    assert menu.checked_items() == []
    menu.toggle()
    assert [i.label for i in menu.checked_items()] == ["a"]
    menu.move(1)
    menu.toggle()
    assert [i.label for i in menu.checked_items()] == ["a", "b"]
    menu.toggle()
    assert [i.label for i in menu.checked_items()] == ["a"]


def test_toggle_noop_when_single_select():
    menu = Menu("m", ["a", "b"])
    watcher = VersionWatcher(menu)
    menu.toggle()
    assert menu.checked_items() == []
    assert menu.version == 0
    assert watcher.calls == 0


def test_accept_single_select():
    menu = Menu("m", [MenuItem("a", value=1), MenuItem("b", value=2)])
    menu.move(1)
    result = menu.accept()
    assert result == 2
    assert menu.done
    assert menu.result == 2


def test_accept_multi_select_checked():
    menu = Menu("m", [MenuItem("a", value=1), MenuItem("b", value=2),
                      MenuItem("c", value=3)], multi_select=True)
    menu.toggle()          # check 'a'
    menu.move(2)
    menu.toggle()          # check 'c'
    result = menu.accept()
    assert result == [1, 3]
    assert menu.result == [1, 3]


def test_accept_multi_select_nothing_checked_falls_back_to_cursor():
    menu = Menu("m", [MenuItem("a", value=1), MenuItem("b", value=2)],
                multi_select=True)
    menu.move(1)
    assert menu.accept() == [2]


def test_cancel():
    menu = Menu("m", ["a"], multi_select=True)
    menu.toggle()
    assert menu.cancel() is None
    assert menu.done
    assert menu.result is None


def test_accept_cancel_idempotent_once_done():
    menu = Menu("m", [MenuItem("a", value=1)])
    assert menu.accept() == 1
    version = menu.version
    assert menu.cancel() == 1      # already done: returns stored result
    assert menu.accept() == 1
    assert menu.version == version


def test_empty_menu():
    single = Menu("m", [])
    watcher = VersionWatcher(single)
    assert single.current is None
    single.move(1)                  # no-op
    single.toggle()                 # no-op
    assert single.version == 0 and watcher.calls == 0
    assert single.accept() is None

    multi = Menu("m", [], multi_select=True)
    assert multi.accept() == []


def test_menu_version_bumps_only_on_real_change():
    menu = Menu("m", ["a", "b"], multi_select=True)
    watcher = VersionWatcher(menu)

    menu.move(0)                    # no change
    assert menu.version == 0 and watcher.calls == 0
    menu.move(-1)                   # clamped, no change
    assert menu.version == 0 and watcher.calls == 0

    menu.move(1)
    assert menu.version == 1 and watcher.calls == 1
    menu.move(5)                    # clamps onto the same index -> no change
    assert menu.version == 1 and watcher.calls == 1

    menu.toggle()
    assert menu.version == 2 and watcher.calls == 2
    menu.accept()
    assert menu.version == 3 and watcher.calls == 3


def test_pager_clamping_and_text():
    pager = Pager("book", ["page one", "page two\nsecond line", "page three"])
    assert pager.page == 0
    assert pager.page_count == 3
    assert pager.page_text == "page one"
    pager.prev_page()
    assert pager.page == 0
    pager.next_page()
    assert pager.page == 1
    assert pager.page_text == "page two\nsecond line"
    pager.next_page()
    pager.next_page()               # clamped at last page
    assert pager.page == 2
    pager.prev_page()
    assert pager.page == 1


def test_empty_pager():
    pager = Pager("book", [])
    assert pager.page_count == 0
    assert pager.page_text == ''
    pager.next_page()
    pager.prev_page()
    assert pager.page == 0 and pager.version == 0


def test_pager_version_and_observer():
    pager = Pager("book", ["a", "b"])
    watcher = VersionWatcher(pager)

    pager.prev_page()               # clamped, no change
    assert pager.version == 0 and watcher.calls == 0
    pager.next_page()
    assert pager.version == 1 and watcher.calls == 1
    pager.next_page()               # clamped, no change
    assert pager.version == 1 and watcher.calls == 1

    pager.close()
    assert pager.done
    assert pager.version == 2 and watcher.calls == 2
    pager.close()                   # idempotent
    assert pager.version == 2 and watcher.calls == 2
