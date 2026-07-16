import numpy as np
import pytest

from carriage_return.layers import FieldLayer, GlyphRegistry, SpriteLayer


# ---------------------------------------------------------------- GlyphRegistry

def test_glyph_registry_ids_are_stable_and_ordered():
    reg = GlyphRegistry()
    first = reg.add_chars('abc')
    assert first == 0
    assert reg['a'] == 0
    assert reg['b'] == 1
    assert reg['c'] == 2
    assert reg.chars == ['a', 'b', 'c']

    second = reg.add_chars(['x', 'y'])
    assert second == 3
    assert reg['x'] == 3
    assert reg.chars[3] == 'x'
    # earlier ids unchanged
    assert reg['a'] == 0
    assert len(reg) == 5


def test_glyph_registry_getitem_adds_missing():
    reg = GlyphRegistry()
    v0 = reg.version
    i = reg['Z']
    assert i == 0
    assert reg.version > v0
    v1 = reg.version
    assert reg['Z'] == 0  # lookup does not re-add
    assert reg.version == v1


# ------------------------------------------------------------------ SpriteLayer

def test_add_sprites_allocates_contiguous_ranges():
    layer = SpriteLayer('test')
    a = layer.add_sprites((3,))
    b = layer.add_sprites((2, 2))
    assert len(layer) == 7
    assert a.indices == (0, 3)
    assert b.indices == (3, 7)
    assert b.position.shape == (2, 2, 3)
    # new sprites start hidden
    assert np.isnan(layer.position).all()


def test_add_sprites_requires_tuple():
    layer = SpriteLayer()
    with pytest.raises(TypeError):
        layer.add_sprites(3)


def test_slot_setters_broadcast():
    layer = SpriteLayer()
    s = layer.add_sprites((2, 3))

    s.position = (1, 2, 3)                    # scalar broadcast
    assert np.all(layer.position[:6] == [1, 2, 3])

    pos = np.zeros((2, 3, 3), dtype='float32')
    pos[..., 0] = np.arange(6).reshape(2, 3)
    s.position = pos                          # full array
    assert np.all(s.position == pos)

    s.glyph = 7
    assert np.all(s.glyph == 7)
    grid = np.arange(6).reshape(2, 3)
    s.glyph = grid
    assert np.all(layer.glyph[:6] == np.arange(6))

    s.fgcolor = (1, 1, 1, 0.5)
    assert np.all(s.fgcolor[..., 3] == 0.5)
    s.bgcolor = None                          # None -> NaN, same as SpriteData
    assert np.isnan(s.bgcolor).all()


def test_version_bumps_on_write_only():
    layer = SpriteLayer()
    s = layer.add_sprites((4,))
    v = layer.version
    sv = layer.structure_version

    s.position = (0, 0, 0)
    assert layer.version > v
    assert layer.structure_version == sv  # plain write does not change structure

    v = layer.version
    _ = s.position  # read
    assert layer.version == v


def test_resize_bumps_structure_version_and_preserves_data():
    layer = SpriteLayer()
    a = layer.add_sprites((2,))
    a.position = [(1, 1, 1), (2, 2, 2)]
    a.glyph = [5, 6]
    a.fgcolor = (1, 0, 0, 1)
    a.bgcolor = (0, 0, 0, 1)

    sv = layer.structure_version
    b = layer.add_sprites((3,))
    assert layer.structure_version > sv
    # a's data survived the reallocation
    assert np.all(a.position == [(1, 1, 1), (2, 2, 2)])
    assert np.all(a.glyph == [5, 6])
    # b hidden until written
    assert np.isnan(b.position).all()


def test_set_shape_repacks_and_reapplies_following_slots():
    layer = SpriteLayer()
    a = layer.add_sprites((2,))
    b = layer.add_sprites((3,))
    b.position = np.array([(1, 1, 1), (2, 2, 2), (3, 3, 3)], dtype='float32')
    b.glyph = np.array([1, 2, 3])
    b.fgcolor = (1, 1, 1, 1)
    b.bgcolor = (0, 0, 0, 1)

    a.set_shape((5,))
    assert len(layer) == 8
    assert a.indices == (0, 5)
    assert b.indices == (5, 8)
    # b's cached data was re-applied at its new offset
    assert np.all(b.position == [(1, 1, 1), (2, 2, 2), (3, 3, 3)])
    assert np.all(b.glyph == [1, 2, 3])
    # a's data is reset by set_shape (same contract as SpriteData); caller rewrites
    assert a._position is None


def test_slot_views_alias_layer_arrays():
    layer = SpriteLayer()
    s = layer.add_sprites((2, 2))
    s.position = 0
    s.position[1, 1] = (9, 9, 9)  # in-place write through the view
    assert np.all(layer.position[3] == 9)


# ------------------------------------------------------------------- FieldLayer

def test_field_layer_set_data_and_bump():
    f = FieldLayer('light', shape=(4, 4, 3))
    assert f.data.dtype == np.dtype('float32')
    v = f.version

    data = np.random.rand(4, 4, 3)
    buf = f.data
    f.set_data(data)
    assert f.version > v
    assert f.data is buf  # same-shape set_data copies in place
    assert np.allclose(f.data, data.astype('float32'))

    f.set_data(np.zeros((8, 8, 3)))  # shape change replaces the array
    assert f.data.shape == (8, 8, 3)

    v = f.version
    f.data[0, 0, 0] = 1.0
    f.bump()
    assert f.version == v + 1


def test_field_layer_from_data():
    src = np.ones((2, 2), dtype='float64')
    f = FieldLayer('opacity', data=src)
    assert f.data.dtype == np.dtype('float32')
    assert np.all(f.data == 1)
