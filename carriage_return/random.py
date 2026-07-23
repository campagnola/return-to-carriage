import numpy as np


class RandomDist:
    """A named, parameterised random distribution you can sample as a callable.

    Built with a distribution name and its parameters under friendly names
    (``value`` for a constant, ``mean``/``stdev`` for the gaussians,
    ``low``/``high`` for uniform, ``n``/``sides`` for dice); the constructor
    validates that exactly the expected parameters are supplied. Call the
    instance for a single scalar sample, or :meth:`sample` for an array. Each
    distribution maps its friendly parameters onto the underlying numpy call
    (which names them differently and takes ``size=`` rather than ``shape=``),
    so callers never see numpy's argument spelling. Use :meth:`update` to retune
    parameters or switch to a different distribution in place -- handy for
    pinning a distribution to a constant in tests, or growing a stat over time.
    """

    #: distribution name -> (sampler method name, required parameter names)
    _DISTS = {
        'constant': ('sample_constant', ['value']),
        'lognorm': ('sample_lognorm', ['mean', 'stdev']),
        'normal': ('sample_normal', ['mean', 'stdev']),
        'uniform': ('sample_uniform', ['low', 'high']),
        'dieroll': ('sample_dieroll', ['n', 'sides']),
        'summed_dieroll': ('sample_summed_dieroll', ['n', 'sides']),
    }

    def __init__(self, dist_type, **params):
        self._configure(dist_type, params)

    def _configure(self, dist_type, params):
        """Validate *dist_type* and *params* and adopt them as this dist's state.

        Requires exactly the parameters the distribution expects -- no missing
        ones, no extras -- so a mistyped parameter is caught rather than silently
        ignored. Shared by __init__ and update, so both reject the same way.
        """
        if dist_type not in self._DISTS:
            raise ValueError(f"Unsupported distribution type: {dist_type}")
        sampler_name, sampler_params = self._DISTS[dist_type]
        for param in sampler_params:
            if param not in params:
                raise ValueError(f"Missing required parameter '{param}' for distribution type '{dist_type}'")
        for param in params:
            if param not in sampler_params:
                raise ValueError(f"Unexpected parameter '{param}' for distribution type '{dist_type}'")
        self.dist_type = dist_type
        self.params = dict(params)
        self.sampler = getattr(self, sampler_name)

    def update(self, **kwds):
        """Retune parameters in place, or switch distribution type.

        Pass ``dist_type=<name>`` to switch distributions; the remaining keywords
        are then that distribution's *full* parameter set. Without ``dist_type``
        the keywords are merged onto the current parameters (a partial update,
        e.g. ``d.update(mean=2)``) and re-validated against the current type.
        Either way the result must be a complete, valid parameter set.
        """
        if 'dist_type' in kwds:
            dist_type = kwds.pop('dist_type')
            params = kwds
        else:
            dist_type = self.dist_type
            params = {**self.params, **kwds}
        self._configure(dist_type, params)

    def __call__(self):
        return self.sample(shape=1)[0]

    def sample(self, shape=1):
        return self.sampler(shape)

    def sample_constant(self, shape):
        return np.full(shape, self.params['value'])

    def sample_lognorm(self, shape):
        return np.random.lognormal(self.params['mean'], self.params['stdev'], size=shape)

    def sample_normal(self, shape):
        return np.random.normal(self.params['mean'], self.params['stdev'], size=shape)

    def sample_uniform(self, shape):
        return np.random.uniform(self.params['low'], self.params['high'], size=shape)

    def sample_dieroll(self, shape):
        n = self.params['n']
        sides = self.params['sides']
        return np.random.randint(1, sides + 1, (n, shape))

    def sample_summed_dieroll(self, shape):
        return np.sum(self.sample_dieroll(shape), axis=0)
