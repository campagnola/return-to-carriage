import numpy as np


def make_noise(shape, rng, spectral_fn, voxel_size_um=1.0, stdev=None):
    """Generate noise with an arbitrary power spectrum via IFFT."""
    ndim = len(shape)
    if np.isscalar(voxel_size_um):
        voxel_size_um = (float(voxel_size_um),) * ndim
    axes_freqs = [np.fft.fftfreq(n, d=d) for n, d in zip(shape, voxel_size_um)]
    grids = np.meshgrid(*axes_freqs, indexing='ij')
    freq_mag = np.sqrt(sum(g ** 2 for g in grids))
    # sqrt(2) compensates for the power lost when taking np.real() of the IFFT:
    # real(IFFT(Z)) averages Z[k] with conj(Z[-k]), halving the power.
    noise_f = np.sqrt(2) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    noise_f *= spectral_fn(freq_mag)
    result = np.real(np.fft.ifftn(noise_f))
    if stdev is not None:
        result = result / result.std() * stdev
    return result


def gaussian_spectrum(sigma_freq, amp=1.0, center_freq=0.0):
    return lambda freq: amp * np.exp(-0.5 * ((freq - center_freq) / sigma_freq) ** 2)


def exponential_spectrum(scale_freq, amp=1.0):
    return lambda freq: amp * np.exp(-freq / scale_freq)


def radial_power_spectrum(image, pixel_um, n_bins=200):
    H, W = image.shape
    psd2d = np.abs(np.fft.fftshift(np.fft.fft2(image))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(H, d=pixel_um))
    fx = np.fft.fftshift(np.fft.fftfreq(W, d=pixel_um))
    FX, FY = np.meshgrid(fx, fy)
    freq_r = np.sqrt(FX ** 2 + FY ** 2)
    freq_max = min(fx.max(), fy.max())
    bin_edges = np.linspace(0, freq_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    spectrum = np.array([
        psd2d[(freq_r >= bin_edges[i]) & (freq_r < bin_edges[i + 1])].mean()
        for i in range(n_bins)
    ])
    return bin_centers, spectrum
