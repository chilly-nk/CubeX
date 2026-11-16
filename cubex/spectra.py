"""Spectral peak measurements."""

import numpy as np

__all__ = ['FWHM']

class FWHM:
    """Measure interpolated full width at half maximum of one peak."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        """Parameters:
        x: 1-D array of wavelengths
        y: 1-D array of intensities. Must correspond to a spectrum with exactly one peak.
        """
        x, y = np.asarray(x), np.asarray(y)
        if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 3:
            raise ValueError('x and y must be matching one-dimensional arrays with at least three samples.')
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or not np.all(np.diff(x) > 0):
            raise ValueError('Samples must be finite and wavelengths strictly increasing.')
        peak_idx = np.argmax(y)
        peak_x = x[peak_idx]
        peak_y = y[peak_idx]
        half_max = peak_y / 2
        left_idx = np.where(y[:peak_idx] < half_max)[0]
        right_idx = np.where(y[peak_idx:] < half_max)[0]
        if not len(left_idx) or not len(right_idx):
            raise ValueError('The spectrum must cross half maximum on both sides of its peak.')
        i1 = left_idx[-1]
        i2 = i1 + 1
        x_left = np.interp(half_max, [y[i1], y[i2]], [x[i1], x[i2]])
        i2 = peak_idx + right_idx[0]
        i1 = i2 - 1
        x_right = np.interp(half_max, [y[i2], y[i1]], [x[i2], x[i1]])
        fwhm = x_right - x_left
        fwhm_center = x_left + fwhm / 2
        peak_offset = fwhm_center - peak_x
        self.x = x
        self.y = y
        self.peak_idx = peak_idx
        self.peak_x = peak_x
        self.peak_y = peak_y
        self.half_max = half_max
        self.left_idx = left_idx
        self.x_left = x_left
        self.right_idx = right_idx
        self.x_right = x_right
        self.fwhm = fwhm
        self.fwhm_center = fwhm_center
        self.peak_offset = peak_offset
