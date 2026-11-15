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


def test_reshape_reallocates_blank_and_bumps_structure(grid, registry):
    grid.set_data([list("hi" + " " * 8)] * 4, grid.fgcolor, grid.bgcolor)
    v, sv = grid.version, grid.structure_version
    grid.reshape((6, 20))
    assert grid.shape == (6, 20)
    assert grid.glyph.shape == (6, 20)
    assert grid.fgcolor.shape == (6, 20, 4)
    assert (grid.glyph == registry[' ']).all()  # contents reset to blank
    assert grid.version > v and grid.structure_version > sv


def test_set_data_glyph_ids_come_from_registry(registry):
    grid = CharGridLayer(registry, (1, 3))
    grid.set_data([['a', 'b', ' ']], grid.fgcolor, grid.bgcolor)
    assert grid.glyph[0, 0] == registry['a']
    assert grid.glyph[0, 1] == registry['b']
    assert chars_of(grid, 0) == 'ab '
    assert grid.version == 1  # one bump for the whole replace


def test_layer_list_membership_and_versioning():
    grids = LayerList()
    calls = []
    grids.changed.connect(lambda: calls.append(grids.structure_version))
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
    layer.changed.connect(lambda: calls.append((layer.version, layer.structure_version)))

    slot = layer.add_sprites((2,))
    assert layer.structure_version == 1
    slot.glyph = 3
    assert layer.version >= 2
    assert calls  # observer fired through the shared _changed()


def test_char_grid_is_glyph_layer(grid):
    assert isinstance(grid, GlyphLayer)
