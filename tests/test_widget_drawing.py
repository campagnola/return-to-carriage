"""Widget drawing primitives: write/fill_row/clear clipping and versioning (headless).

These moved from CharGridLayer to Widget when the widget tree took over
drawing; CharGridLayer now only takes bulk set_data() from the compositor.
"""
import numpy as np
import pytest

from carriage_return.widgets import BG, FG, Widget


@pytest.fixture
def widget():
    return Widget(4, 10)


def chars_of(widget, row):
    return ''.join(widget.glyph[row])


def test_new_widget_is_blank_spaces(widget):
    assert widget.shape == (4, 10)
    assert (widget.glyph == ' ').all()
    assert (widget.fgcolor == np.float32(FG)).all()
    assert (widget.bgcolor == np.float32(BG)).all()
    assert widget.version == 0


def test_write_and_decode(widget):
    widget.write(1, 2, "hi!")
    assert chars_of(widget, 1) == '  hi!     '
    assert widget.version == 1  # one bump for the whole write


def test_write_colors_only_written_cells(widget):
    widget.write(0, 0, "ab", fg=(1, 0, 0, 1), bg=(0, 0, 1, 1))
    assert (widget.fgcolor[0, :2] == (1, 0, 0, 1)).all()
    assert (widget.fgcolor[0, 2:] == np.float32(FG)).all()
    assert (widget.bgcolor[0, :2] == (0, 0, 1, 1)).all()
    assert (widget.bgcolor[0, 2:] == np.float32(BG)).all()


def test_write_clips_right_edge(widget):
    widget.write(0, 8, "abcdef")
    assert chars_of(widget, 0) == '        ab'
    assert widget.version == 1


def test_write_clips_left_edge(widget):
    widget.write(0, -2, "abcdef")
    assert chars_of(widget, 0) == 'cdef      '


def test_fully_clipped_write_changes_nothing(widget):
    widget.write(7, 0, "off the grid")   # row out of range
    widget.write(0, 10, "too far right")
    widget.write(0, -20, "gone entirely")
    assert widget.version == 0


def test_fill_row_recolors_without_touching_glyphs(widget):
    widget.write(2, 0, "text")
    version = widget.version
    widget.fill_row(2, fg=(0, 0, 0, 1), bg=(1, 1, 0, 1))
    assert chars_of(widget, 2).startswith('text')
    assert (widget.fgcolor[2] == (0, 0, 0, 1)).all()
    assert (widget.bgcolor[2] == (1, 1, 0, 1)).all()
    assert widget.version == version + 1


def test_clear_resets_cells(widget):
    widget.write(0, 0, "junk", fg=(1, 0, 0, 1))
    widget.clear(fg=(0.5, 0.5, 0.5, 1), bg=(0, 0, 0, 0.9))
    assert (widget.glyph == ' ').all()
    assert (widget.fgcolor == np.float32((0.5, 0.5, 0.5, 1))).all()
    assert (widget.bgcolor == np.float32((0, 0, 0, 0.9))).all()


def test_observer_invoked_per_change(widget):
    calls = []
    widget.changed.connect(lambda: calls.append(widget.version))
    widget.write(0, 0, "a")
    widget.fill_row(0, fg=(1, 1, 1, 1))
    widget.clear()
    assert calls == [1, 2, 3]
