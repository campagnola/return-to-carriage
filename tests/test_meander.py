"""The wandering-curve generators in carriage_return.terrain.meander."""
import numpy as np
import pytest

from carriage_return.terrain.meander import meander, natural_extent


# -- meander() -----------------------------------------------------------

def test_meander_lands_exactly_on_endpoints():
    rng = np.random.RandomState(0)
    v = meander(rng, 50, start_center=5, end_center=20, amplitude=3)
    assert v[0] == 5
    assert v[-1] == 20


def test_meander_length_matches():
    rng = np.random.RandomState(0)
    v = meander(rng, 37, start_center=0, end_center=0, amplitude=2)
    assert len(v) == 37


# -- natural_extent() -----------------------------------------------------

@pytest.mark.parametrize('max_extent', [0, 1, 3, 8])
def test_natural_extent_stays_in_bounds(max_extent):
    rng = np.random.RandomState(1)
    v = natural_extent(rng, 200, max_extent)
    assert v.min() >= 0
    assert v.max() <= max_extent


def test_natural_extent_length_matches():
    rng = np.random.RandomState(1)
    v = natural_extent(rng, 123, 3)
    assert len(v) == 123


def test_natural_extent_is_integer_dtype():
    rng = np.random.RandomState(1)
    v = natural_extent(rng, 50, 3)
    assert np.issubdtype(v.dtype, np.integer)


def test_natural_extent_zero_max_is_all_zero():
    rng = np.random.RandomState(1)
    v = natural_extent(rng, 50, 0)
    assert np.all(v == 0)


@pytest.mark.parametrize('max_extent', [1, 3])
def test_natural_extent_visits_multiple_values(max_extent):
    # a small max_extent shouldn't pin at one end or flatline -- it should
    # actually spend time at more than one distinct value along a long run.
    rng = np.random.RandomState(2)
    v = natural_extent(rng, 300, max_extent)
    assert len(np.unique(v)) > 1


@pytest.mark.parametrize('max_extent', [1, 3])
def test_natural_extent_is_spatially_correlated_not_iid_noise(max_extent):
    # same-value runs are typically longer than 1 cell, unlike per-cell
    # independent noise (which would mostly produce runs of length 1).
    rng = np.random.RandomState(3)
    v = natural_extent(rng, 300, max_extent)
    runs = []
    run_len = 1
    for i in range(1, len(v)):
        if v[i] == v[i - 1]:
            run_len += 1
        else:
            runs.append(run_len)
            run_len = 1
    runs.append(run_len)
    assert np.mean(runs) > 2.0


@pytest.mark.parametrize('max_extent', [1, 3])
def test_natural_extent_adjacent_jumps_are_rarely_large(max_extent):
    rng = np.random.RandomState(4)
    v = natural_extent(rng, 300, max_extent)
    jumps = np.abs(np.diff(v))
    assert np.mean(jumps > 1) < 0.1


def test_natural_extent_independent_calls_differ():
    # the same rng, called twice in a row (as steps 5/6 will for e.g. left
    # vs. right bank width), should produce different sequences since the
    # rng's state has advanced between calls.
    rng = np.random.RandomState(5)
    a = natural_extent(rng, 100, 3)
    b = natural_extent(rng, 100, 3)
    assert not np.array_equal(a, b)
