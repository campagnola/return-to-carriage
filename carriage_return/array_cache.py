

class ArraySumCache:
    """For efficiently adding multiple arrays together, in cases where the same set of arrays will be added
    repeatedly plus a smaller number of arrays that change each time.


    """

    def __init__(self):
        self._last_arrays = set()  # all arrays requested for summation on the previous round
        self._last_cache_arrays = set()  # arrays used for cache sum 
        self._last_cache = None  # last cache sum result
        self._last_array_ids = None  # temporarily store ref to arrays to prevent ID reuse

        # test metrics
        self._n_used_from_cache = None
        self._n_summed = None
        self._n_stored_in_cache = None

    def sum_arrays(self, array_list):
        assert len(array_list) > 0

        array_ids = {id(a):a for a in array_list}
        self._last_array_ids = array_ids

        arrays = set(array_ids.keys())

        # if all arrays from previous cache are being summed again, then
        # reuse the previously cached result
        repeated_from_cache = arrays & self._last_cache_arrays
        if repeated_from_cache == self._last_cache_arrays:
            # previous cache is still valid, start there
            self._n_used_from_cache = len(repeated_from_cache)
            total = self._last_cache
            summed_arrays = repeated_from_cache
        else:
            self._n_used_from_cache = 0
            total = None
            summed_arrays = set()

        # Of the remaining data to be summed, anything that was repeated since last call
        # can be added in to the new cache
        remaining_arrays = arrays - summed_arrays
        repeated = remaining_arrays & self._last_arrays

        for arr in repeated:
            # add remaining data that was repeated from last time, cache new result
            if total is None:
                total = array_ids[arr].copy()
            else:
                total += array_ids[arr]
        summed_arrays = summed_arrays | repeated
        remaining_arrays = remaining_arrays - repeated

        self._last_cache = None if total is None else total.copy()
        self._last_cache_arrays = summed_arrays.copy()
        self._n_stored_in_cache = len(summed_arrays)

        # add in any remaining new data
        for arr in remaining_arrays:
            if total is None:
                total = array_ids[arr].copy()
            else:
                total += array_ids[arr]
        
        self._n_summed = len(repeated) + len(remaining_arrays)

        self._last_arrays = arrays

        return total



