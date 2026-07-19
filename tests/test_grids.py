"""CharGridLayer / LayerList / shared GlyphLayer contract (headless)."""
import numpy as np
import pytest

from carriage_return.layers import (GlyphLayer, GlyphRegistry, CharGridLayer,
                                    LayerList, SpriteLayer)


@pytest.fixture
def registry():
    return GlyphRegistry()


@pytest.fixture
def grid(registry):
    return CharGridLayer(registry, (4, 10))


def chars_of(grid, row):
    """Decode a grid row back to a string via its registry."""
    return ''.join(grid.registry.chars[i] for i in grid.glyph[row])


def test_new_grid_is_blank_spaces(grid, registry):
    assert grid.shape == (4, 10)
    assert (grid.glyph == registry[' ']).all()
    assert (grid.fgcolor == 1.0).all()
    assert (grid.bgcolor == 0.0).all()
    assert grid.space == 'screen' and grid.anchor == 'center'
    assert grid.version == 0 and grid.structure_version == 0


def test_write_and_decode(grid):
    grid.write(1, 2, "hi!")
    assert chars_of(grid, 1) == '  hi!     '
    assert grid.version == 1  # one bump for the whole write


def test_write_colors_only_written_cells(grid):
    grid.write(0, 0, "ab", fg=(1, 0, 0, 1), bg=(0, 0, 1, 1))
    assert (grid.fgcolor[0, :2] == (1, 0, 0, 1)).all()
    assert (grid.fgcolor[0, 2:] == 1.0).all()
    assert (grid.bgcolor[0, :2] == (0, 0, 1, 1)).all()
    assert (grid.bgcolor[0, 2:] == 0.0).all()


def test_reshape_reallocates_blank_and_bumps_structure(grid, registry):
    grid.write(0, 0, "hi")
    v, sv = grid.version, grid.structure_version
    grid.reshape((6, 20))
    assert grid.shape == (6, 20)
    assert grid.glyph.shape == (6, 20)
    assert grid.fgcolor.shape == (6, 20, 4)
    assert (grid.glyph == registry[' ']).all()  # contents reset to blank
    assert grid.version > v and grid.structure_version > sv


def test_write_clips_right_edge(grid):
    grid.write(0, 8, "abcdef")
    assert chars_of(grid, 0) == '        ab'
    assert grid.version == 1


def test_write_clips_left_edge(grid):
    grid.write(0, -2, "abcdef")
    assert chars_of(grid, 0) == 'cdef      '


def test_fully_clipped_write_changes_nothing(grid):
    grid.write(7, 0, "off the grid")   # row out of range
    grid.write(0, 10, "too far right")
    grid.write(0, -20, "gone entirely")
    assert grid.version == 0


def test_fill_row_recolors_without_touching_glyphs(grid):
    grid.write(2, 0, "text")
    version = grid.version
    grid.fill_row(2, fg=(0, 0, 0, 1), bg=(1, 1, 0, 1))
    assert chars_of(grid, 2).startswith('text')
    assert (grid.fgcolor[2] == (0, 0, 0, 1)).all()
    assert (grid.bgcolor[2] == (1, 1, 0, 1)).all()
    assert grid.version == version + 1


def test_clear_resets_cells(grid, registry):
    grid.write(0, 0, "junk", fg=(1, 0, 0, 1))
    grid.clear(fg=(0.5, 0.5, 0.5, 1), bg=(0, 0, 0, 0.9))
    assert (grid.glyph == registry[' ']).all()
    assert (grid.fgcolor == np.float32((0.5, 0.5, 0.5, 1))).all()
    assert (grid.bgcolor == np.float32((0, 0, 0, 0.9))).all()


def test_observer_invoked_per_change(grid):
    calls = []
    grid.observer = lambda: calls.append(grid.version)
    grid.write(0, 0, "a")
    grid.fill_row(0, fg=(1, 1, 1, 1))
    grid.clear()
    assert calls == [1, 2, 3]


def test_glyph_ids_come_from_registry(registry):
    grid = CharGridLayer(registry, (1, 3))
    grid.write(0, 0, "ab")
    assert grid.glyph[0, 0] == registry['a']
    assert grid.glyph[0, 1] == registry['b']


def test_layer_list_membership_and_versioning():
    grids = LayerList()
    calls = []
    grids.observer = lambda: calls.append(grids.structure_version)
    registry = GlyphRegistry()
    a = CharGridLayer(registry, (1, 1))
    b = CharGridLayer(registry, (1, 1))

    grids.add(a)
    grids.add(b)
    assert list(grids) == [a, b] and len(grids) == 2
    assert grids[1] is b
    assert grids.structure_version == 2

    grids.remove(a)
    assert list(grids) == [b]
    assert grids.structure_version == 3
    assert calls == [1, 2, 3]

    with pytest.raises(ValueError):
        grids.remove(a)


def test_sprite_layer_shares_glyph_layer_contract():
    layer = SpriteLayer('actors')
    assert isinstance(layer, GlyphLayer)
    calls = []
    layer.observer = lambda: calls.append((layer.version, layer.structure_version))

    slot = layer.add_sprites((2,))
    assert layer.structure_version == 1
    slot.glyph = 3
    assert layer.version >= 2
    assert calls  # observer fired through the shared _changed()


def test_char_grid_is_glyph_layer(grid):
    assert isinstance(grid, GlyphLayer)
