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
