from __future__ import annotations
from dataclasses import dataclass, replace
import math
import numpy as np
from scipy import stats


def _rng(g: np.random.Generator | None):
    """*g*, or the global ``numpy.random`` state if none was given."""
    return g if g is not None else np.random


@dataclass
class Dist:
    """A 1-D random parameter with a selectable distribution.

    One tagged dataclass (rather than a class hierarchy) so it serializes
    trivially and maps directly onto a GUI's DistParameter-style widget.
    ``kind`` selects which fields are live:

      * ``uniform``   -> ``lo``, ``hi``           : lo + (hi-lo)*U
      * ``normal``    -> ``mu``, ``sigma``        : mu + sigma*N(0,1)
      * ``lognormal`` -> ``mu``, ``sigma``        : exp(mu + sigma*N(0,1))
      * ``binomial``  -> ``n``, ``p``             : Binomial(n, p) -> int

    For lognormal, ``mu`` is the underlying-normal mean, i.e. the log of the
    median; the GUI shows/edits the median directly and converts.

    ``clip_min``/``clip_max`` cause out-of-bounds draws to be resampled, keeping
    the distribution shape intact. ``integer`` rounds the result (used for the
    layer count). ``dtype``, if set, casts each sample via ``dtype(v)``; use
    ``.astype()`` to build a coerced copy.
    """
    kind: str = "uniform"
    lo: float = 0.0
    hi: float = 1.0
    mu: float = 0.0
    sigma: float = 1.0
    n: int = 1        # binomial: number of trials
    p: float = 0.5    # binomial: success probability
    clip_min: float | None = None
    clip_max: float | None = None
    integer: bool = False
    dtype: type | None = None

    def _scipy_dist(self):
        """The frozen scipy.stats distribution backing this ``kind``."""
        if self.kind == "uniform":
            return stats.uniform(loc=self.lo, scale=self.hi - self.lo)
        if self.kind == "normal":
            return stats.norm(loc=self.mu, scale=self.sigma)
        if self.kind == "lognormal":
            return stats.lognorm(s=self.sigma, scale=math.exp(self.mu))
        if self.kind == "binomial":
            return stats.binom(n=self.n, p=self.p)
        raise ValueError(f"unknown distribution kind {self.kind!r}")

    def sample(self, n: int = 1, g: np.random.Generator | None = None) -> float | int | list:
        if n != 1:
            return [self.sample(g=g) for _ in range(n)]
        d = self._scipy_dist()
        rng = _rng(g)
        while True:
            v = d.rvs(random_state=rng)
            if (self.clip_min is None or v >= self.clip_min) and (self.clip_max is None or v <= self.clip_max):
                if self.kind == "binomial":
                    result = int(v)
                elif self.integer:
                    result = int(round(v))
                else:
                    result = float(v)
                return self.dtype(result) if self.dtype is not None else result

    def _clip_norm(self) -> float:
        """CDF mass in [clip_min, clip_max] -- normalization constant for the truncated distribution."""
        d = self._scipy_dist()
        if self.kind == "binomial":
            k_lo = 0 if self.clip_min is None else max(0, math.ceil(self.clip_min))
            k_hi = self.n if self.clip_max is None else min(self.n, math.floor(self.clip_max))
            return float(d.cdf(k_hi) - d.cdf(k_lo - 1))
        lo = self.clip_min if self.clip_min is not None else -math.inf
        hi = self.clip_max if self.clip_max is not None else math.inf
        return float(d.cdf(hi) - d.cdf(lo))

    def pdf(self, x):
        """Probability density (or mass for binomial) at x; x may be scalar or array."""
        scalar = np.ndim(x) == 0
        x = np.atleast_1d(np.asarray(x, dtype=float))
        d = self._scipy_dist()
        p = d.pmf(x) if self.kind == "binomial" else d.pdf(x)

        if self.clip_min is not None or self.clip_max is not None:
            lo = self.clip_min if self.clip_min is not None else -math.inf
            hi = self.clip_max if self.clip_max is not None else math.inf
            p = np.where((x >= lo) & (x <= hi), p, 0.0)
            norm = self._clip_norm()
            if norm > 0:
                p = p / norm

        return float(p[0]) if scalar else p

    def __add__(self, other):
        if isinstance(other, Dist):
            left = self.dists if isinstance(self, MixedDist) else [self]
            right = other.dists if isinstance(other, MixedDist) else [other]
            return MixedDist(left + right)
        return ModifiedDist(self, 'shift', float(other))

    def __radd__(self, other):
        return ModifiedDist(self, 'shift', float(other))

    def __mul__(self, other):
        return ModifiedDist(self, 'scale', float(other))

    def __rmul__(self, other):
        return ModifiedDist(self, 'scale', float(other))

    def __neg__(self):
        return ModifiedDist(self, 'scale', -1.0)

    def astype(self, dtype: type) -> Dist:
        """Return a copy of this distribution whose samples are cast to *dtype*."""
        return replace(self, dtype=dtype)

    def label(self) -> str:
        """Compact human-readable summary for the decisions printout."""
        if self.kind == "uniform":
            return f"U[{self.lo:.3g},{self.hi:.3g}]"
        if self.kind == "normal":
            return f"N(mu={self.mu:.3g},sig={self.sigma:.3g})"
        if self.kind == "lognormal":
            return f"logN(med={math.exp(self.mu):.3g},sig={self.sigma:.3g})"
        if self.kind == "binomial":
            return f"Bin(n={self.n},p={self.p:.3g})"
        return self.kind


def uniform(lo: float, hi: float, **kw) -> Dist:
    return Dist(kind="uniform", lo=lo, hi=hi, **kw)


def normal(mu: float, sigma: float, **kw) -> Dist:
    return Dist(kind="normal", mu=mu, sigma=sigma, **kw)


def lognormal(median: float = None, sigma: float = 1.0, *, peak: float = None, **kw) -> Dist:
    """Lognormal distribution. Specify either *median* or *peak* (mode), not both."""
    assert (median is None) != (peak is None), "specify exactly one of median or peak"
    mu = math.log(median) if median is not None else math.log(peak) + sigma ** 2
    return Dist(kind="lognormal", mu=mu, sigma=sigma, **kw)


def binomial(n: int, p: float, **kw) -> Dist:
    """Binomial distribution; samples are ints in [0, n]."""
    return Dist(kind="binomial", n=n, p=p, **kw)


class RandomChoice(Dist):
    """Sample from one of several distributions, chosen with given probabilities."""

    def __init__(self, choices: list, weights: list):
        assert len(choices) == len(weights), "choices and weights must have the same length"
        super().__init__(kind="random_choice")
        total = sum(weights)
        self.choices = choices
        self.cumulative = [sum(weights[:i+1]) / total for i in range(len(weights))]

    def sample(self, n: int = 1, g: np.random.Generator | None = None) -> float | list:
        if n != 1:
            return [self.sample(g=g) for _ in range(n)]
        u = _rng(g).random()
        for dist, threshold in zip(self.choices, self.cumulative):
            if u < threshold:
                return dist.sample(g=g) if hasattr(dist, 'sample') else float(dist)
        return self.choices[-1].sample(g=g) if hasattr(self.choices[-1], 'sample') else float(self.choices[-1])

    def pdf(self, x):
        """Weighted mixture of constituent PDFs."""
        scalar = np.ndim(x) == 0
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        weights = [self.cumulative[0]] + [
            self.cumulative[i] - self.cumulative[i - 1] for i in range(1, len(self.cumulative))
        ]
        result = np.zeros_like(x_arr)
        for dist, w in zip(self.choices, weights):
            if hasattr(dist, 'pdf'):
                result += w * np.atleast_1d(np.asarray(dist.pdf(x_arr), dtype=float))
        return float(result[0]) if scalar else result

    def label(self) -> str:
        labels = [d.label() if hasattr(d, 'label') else str(d) for d in self.choices]
        return f"choice({', '.join(labels)})"


def random_choice(choices: list, weights: list) -> RandomChoice:
    return RandomChoice(choices=choices, weights=weights)


class ModifiedDist(Dist):
    """A distribution transformed by a shift (+) or scale (*) operation."""

    def __init__(self, dist: Dist, op: str, operand: float):
        super().__init__(kind="modified")
        self.dist = dist
        self.op = op   # 'shift' or 'scale'
        self.operand = operand

    def sample(self, n: int = 1, g: np.random.Generator | None = None) -> float | list:
        if n != 1:
            return [self.sample(g=g) for _ in range(n)]
        v = self.dist.sample(g=g)
        if self.op == 'shift':
            return v + self.operand
        return v * self.operand

    def pdf(self, x):
        scalar = np.ndim(x) == 0
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        if self.op == 'shift':
            # f_Y(y) = f_X(y - b)  where Y = X + b
            p = np.atleast_1d(np.asarray(self.dist.pdf(x_arr - self.operand), dtype=float))
        else:
            # f_Y(y) = f_X(y/c) / |c|  where Y = X * c
            p = np.atleast_1d(np.asarray(self.dist.pdf(x_arr / self.operand), dtype=float)) / abs(self.operand)
        return float(p[0]) if scalar else p

    def label(self) -> str:
        base = self.dist.label()
        if self.op == 'shift':
            sign = '+' if self.operand >= 0 else ''
            return f"({base}{sign}{self.operand:g})"
        return f"({base}*{self.operand:g})"


class MixedDist(Dist):
    """Equal-weight mixture of distributions; pdf is the mean of constituent pdfs."""

    def __init__(self, dists: list):
        super().__init__(kind="mixed")
        self.dists = list(dists)

    def sample(self, n: int = 1, g: np.random.Generator | None = None) -> float | list:
        if n != 1:
            return [self.sample(g=g) for _ in range(n)]
        idx = int(_rng(g).random() * len(self.dists))
        return self.dists[idx].sample(g=g)

    def pdf(self, x):
        scalar = np.ndim(x) == 0
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        result = np.zeros_like(x_arr)
        for d in self.dists:
            result += np.atleast_1d(np.asarray(d.pdf(x_arr), dtype=float))
        result /= len(self.dists)
        return float(result[0]) if scalar else result

    def label(self) -> str:
        return f"mixed({', '.join(d.label() for d in self.dists)})"
