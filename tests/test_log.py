"""MessageLog: game-state message channel (headless)."""
import pytest

from carriage_return.scene import DEFAULT_SALIENCE, LogEntry, MessageLog


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


def test_entries_carry_salience_and_turn():
    log = MessageLog()
    log.write("early", salience=0.9)
    log.advance_turn()
    log.write("late")
    assert log.entries == [LogEntry("early", 0.9, 0), LogEntry("late", DEFAULT_SALIENCE, 1)]


def test_multiline_write_stamps_every_line():
    log = MessageLog()
    log.advance_turn()
    log.write("a\nb", salience=0.2)
    assert log.entries == [LogEntry("a", 0.2, 1), LogEntry("b", 0.2, 1)]


def test_set_last_line_keeps_salience_and_turn():
    log = MessageLog()
    log.write("> _", salience=1.0)
    log.advance_turn()
    log.set_last_line("> t_")
    assert log.entries == [LogEntry("> t_", 1.0, 0)]


def test_advance_turn_bumps_version_and_fires_changed():
    log = MessageLog()
    calls = []
    log.changed.connect(lambda: calls.append(log.version))
    log.advance_turn()
    assert (log.turn, log.version, calls) == (1, 1, [1])


@pytest.mark.parametrize("salience", [-0.1, 1.1, 10.0])
def test_out_of_range_salience_raises(salience):
    log = MessageLog()
    with pytest.raises(ValueError):
        log.write("x", salience=salience)
    assert log.entries == []
