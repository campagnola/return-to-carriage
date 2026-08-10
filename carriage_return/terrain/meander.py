"""The wandering-curve generator shared by every meandering terrain feature
(dirt paths, rivers -- see :mod:`.path`) so they all read as the same kind of
organic shape rather than each rolling its own.
"""
import numpy as np


def meander(rng, length, start_center, end_center, amplitude, wavelength=25):
    """An organic, wandering integer curve of *length* values, drifting from
    *start_center* to *end_center*.

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
    return np.round(v).astype(int)
