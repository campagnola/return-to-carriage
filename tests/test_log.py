"""MessageLog: game-state message channel (headless)."""
import pytest

from carriage_return.scene import MessageLog


def test_write_appends_and_splits():
    log = MessageLog()
    log.write("hello")
    log.write("two\nlines")
    assert log.lines == ["hello", "two", "lines"]
    assert log.version == 2  # one bump per write, not per line


def test_last_line_editing():
    log = MessageLog()
    log.write("> _")
    log.set_last_line("> t_")
    assert log.lines == ["> t_"]
    log.remove_last_line()
    assert log.lines == []
    assert log.version == 3


def test_observer_fires_per_mutation():
    log = MessageLog()
    calls = []
    log.changed.connect(lambda: calls.append(log.version))
    log.write("a")
    log.set_last_line("b")
    log.remove_last_line()
    assert calls == [1, 2, 3]


def test_editing_empty_log_raises():
    # no defensive silencing: editing a line that doesn't exist is a bug
    log = MessageLog()
    with pytest.raises(IndexError):
        log.set_last_line("x")
    with pytest.raises(IndexError):
        log.remove_last_line()
