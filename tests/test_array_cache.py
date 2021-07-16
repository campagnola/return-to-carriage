import numpy as np
from carriage_return.array_cache import ArraySumCache


def test_arraysumcache():
    cache = ArraySumCache()

    array_list = list(np.random.normal(size=(5, 10, 10)))

    result = cache.sum_arrays(array_list)
    assert np.allclose(result, sum(array_list))
    assert cache._n_used_from_cache == 0
    assert cache._n_stored_in_cache == 0
    assert cache._n_summed == 5

    array_list.append(np.random.normal(size=(10, 10)))

    result = cache.sum_arrays(array_list)
    assert np.allclose(result, sum(array_list))
    assert cache._n_used_from_cache == 0
    assert cache._n_stored_in_cache == 5
    assert cache._n_summed == 6

    array_list[5] = np.random.normal(size=(10, 10))

    result = cache.sum_arrays(array_list)
    assert np.allclose(result, sum(array_list))
    assert cache._n_used_from_cache == 5
    assert cache._n_stored_in_cache == 5
    assert cache._n_summed == 1


