"""Spatial and spectral binning helpers."""

import numpy as np

__all__ = ['bin_cube', 'bin_wavelengths', 'unbin_coord', 'bin_coord', 'bin_bands']

def bin_cube(cube, y_factor=1, x_factor=1, lambda_factor=1):
    """Bin a hyperspectral cube by averaging.

    Parameters
    ----------
    cube : ndarray
        Input cube of shape (y, x, wavelength).
    y_factor : int
        Binning factor along y.
    x_factor : int
        Binning factor along x.
    lambda_factor : int
        Binning factor along wavelength.

    Returns
    -------
    ndarray
        Binned cube.
    """
    for factor in (y_factor, x_factor, lambda_factor):
        if not isinstance(factor, (int, np.integer)) or factor < 1:
            raise ValueError('Binning factors must be positive integers.')
    y, x, l = cube.shape
    if y % y_factor != 0:
        raise ValueError(f'y dimension ({y}) is not divisible by {y_factor}')
    if x % x_factor != 0:
        raise ValueError(f'x dimension ({x}) is not divisible by {x_factor}')
    if l % lambda_factor != 0:
        raise ValueError(f'wavelength dimension ({l}) is not divisible by {lambda_factor}')
    return cube.reshape(y // y_factor, y_factor, x // x_factor, x_factor, l // lambda_factor, lambda_factor).mean(axis=(1, 3, 5))

def bin_wavelengths(wavelengths, lambda_factor=1):
    """Get wavelengths corresponding to spectral binning.

    Parameters
    ----------
    wavelengths : array-like
        Original wavelength values.
    lambda_factor : int
        Binning factor along the wavelength dimension.

    Returns
    -------
    ndarray
        Binned wavelength values, calculated as the mean
        wavelength of each bin.
    """
    if not isinstance(lambda_factor, (int, np.integer)) or lambda_factor < 1:
        raise ValueError('Binning factor must be a positive integer.')
    wavelengths = np.asarray(wavelengths)
    l = len(wavelengths)
    if l % lambda_factor != 0:
        raise ValueError(f'wavelength dimension ({l}) is not divisible by {lambda_factor}')
    return wavelengths.reshape(l // lambda_factor, lambda_factor).mean(axis=1)

def unbin_coord(x, current_binning, target_binning):
    """Map a binned coordinate to a half-open target interval."""
    f = current_binning / target_binning
    x_start = x * f + 0
    x_end = x * f + f
    return (int(x_start), int(x_end))

def bin_coord(x, current_binning, target_binning):
    """Map a coordinate between absolute binning factors."""
    return int(x * current_binning / target_binning)

def bin_bands(color_bands, current_binning, target_binning):
    """Map RGB band indices between absolute binning factors."""
    bands = color_bands.copy()
    b1 = bin_coord(bands['blue'][0], current_binning=current_binning, target_binning=target_binning)
    b2 = bin_coord(bands['blue'][1], current_binning=current_binning, target_binning=target_binning)
    g1 = bin_coord(bands['green'][0], current_binning=current_binning, target_binning=target_binning)
    g2 = bin_coord(bands['green'][1], current_binning=current_binning, target_binning=target_binning)
    r1 = bin_coord(bands['red'][0], current_binning=current_binning, target_binning=target_binning)
    r2 = bin_coord(bands['red'][1], current_binning=current_binning, target_binning=target_binning)
    bands = {'blue': (b1, b2), 'green': (g1, g2), 'red': (r1, r2)}
    return bands
