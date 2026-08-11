"""The wandering-curve generator shared by every meandering terrain feature
(dirt paths, rivers -- see :mod:`.path`) so they all read as the same kind of
organic shape rather than each rolling its own.
"""
import numpy as np


def meander(rng, length, start_center, end_center, amplitude, wavelength=25):
    """An organic, wandering integer curve of *length* values, running
    exactly from *start_center* to *end_center*.

    A mean-reverting random walk -- Brownian motion pulled back toward a
    centreline by an attractor, i.e. an Ornstein-Uhlenbeck process -- then
    lightly smoothed. Unlike linear interpolation between sparse control
    points (which lays down dead-straight segments between them), this never
    stops curving, so it reads as a natural meander rather than a zig-zag.
    The centreline itself drifts linearly from *start_center* to
    *end_center* over the curve's length (a fixed centre, for a path that
    doesn't drift, is just the special case where the two are equal);
    *amplitude* is roughly the typical wander distance from that centreline,
    *wavelength* is roughly the distance it takes to turn.
    """
    theta = 1.0 / wavelength
    sigma = amplitude * np.sqrt(2 * theta)
    steps = rng.normal(0.0, sigma, size=length)
    center = np.linspace(start_center, end_center, length)
    v = np.empty(length)
    v[0] = center[0]
    for i in range(1, length):
        v[i] = v[i - 1] + theta * (center[i] - v[i - 1]) + steps[i]

    # a light box-smooth rounds single-cell jitter into a soft, organic curve
    k = max(3, wavelength // 6)
    if k % 2 == 0:
        k += 1
    padded = np.pad(v, (k // 2, k // 2), mode='edge')
    v = np.convolve(padded, np.ones(k) / k, mode='valid')

    # The walk above only pulls loosely toward start_center/end_center --
    # theta is weak by design, for a natural-looking wander -- and the box
    # smooth above perturbs both ends again regardless. A caller that needs
    # the curve to actually land on end_center (e.g. two path halves meeting
    # a bridge at the same coordinate) can't rely on either. Corrected here
    # the way a Brownian bridge is built from plain Brownian motion: spread
    # each end's residual error across the whole curve as a linear ramp, so
    # the curve settles smoothly into its endpoints instead of jumping to
    # them on the last cell.
    ramp = np.linspace(0.0, 1.0, length)
    v += (start_center - v[0]) * (1.0 - ramp) + (end_center - v[-1]) * ramp

    return np.round(v).astype(int)


def natural_extent(rng, length, max_extent, wavelength=10):
    """A smoothly-varying sequence of *length* small non-negative integers,
    each in ``[0, max_extent]``, for things like a bank or greenery band
    whose width drifts along a river or path rather than jumping around
    per-cell.

    Built the same way :func:`create_river` varies its own width: a
    :func:`meander` run wandering around a fixed centreline
    (``max_extent / 2``) rather than drifting between two endpoints, clipped
    into range afterward. There's no "must land exactly here" requirement
    the way a path meeting a bridge has, so :func:`meander`'s Brownian-bridge
    endpoint correction is along for the ride but harmless -- both ends sit
    at the same centre anyway.

    *amplitude* is fixed at *max_extent* itself (rather than exposed as a
    parameter) so the walk actually reaches both 0 and *max_extent* instead
    of hugging the middle -- this is a short-range quantity (typically 1-3),
    not a long wander like a river's centerline, so there's little reason to
    tune it per call. *wavelength* keeps its default low relative to
    :func:`meander`'s own (25) so the extent actually varies a few times
    over a river's length instead of flatlining at one value.

    Independent calls with the same *rng* (e.g. one for each side of a
    river, one for banks and another for greenery) naturally produce
    different sequences, since the RNG's state has moved on between calls.
    """
    if max_extent <= 0:
        return np.zeros(length, dtype=int)
    center = max_extent / 2.0
    v = meander(rng, length, center, center, amplitude=max_extent, wavelength=wavelength)
    return np.clip(v, 0, max_extent)
