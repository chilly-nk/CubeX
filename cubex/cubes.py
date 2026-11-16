"""Hyperspectral cube loading and exploration."""

from pathlib import Path

import numpy as np
import spectral as spy
from PIL import Image
import matplotlib.pyplot as plt

from .preprocessing import bin_cube, bin_wavelengths, bin_bands
from .visualization import get_rgb

__all__ = ['Cube', 'read_cube_envi', 'read_wvls_envi', 'read_fwhm_envi']

class Cube:
    """Load and work with a Specim ENVI hyperspectral cube."""

    def __init__(self, data_path, origin='specim'):
        """Initialize the object from the supplied data."""
        self.data_path = Path(data_path)
        self.name = self.data_path.stem
        if origin != 'specim':
            raise ValueError("Only origin='specim' is supported.")
        self.processed = None
        self.rgb = None
        self.current_binning = 1
        if origin == 'specim':
            self.raw = read_cube_envi(data_path, format='raw')
            self.wvls = np.array(read_wvls_envi(data_path))
            self.fwhm = np.array(read_fwhm_envi(data_path))
            self.color_bands = {'blue': (33, 49), 'green': (109, 125), 'red': (183, 199)}

    def bin(self, y=1, x=1, wvl=1):
        """Average spatial and spectral bins in place; return this cube."""
        self.raw = bin_cube(self.raw, y_factor=y, x_factor=x, lambda_factor=wvl)
        if self.processed is not None:
            self.processed = bin_cube(self.processed, y_factor=y, x_factor=x, lambda_factor=wvl)
        if wvl != 1:
            self.wvls = bin_wavelengths(self.wvls, lambda_factor=wvl)
            self.fwhm = bin_wavelengths(self.fwhm, lambda_factor=wvl)
            target_binning = self.current_binning * wvl
            self.color_bands = bin_bands(self.color_bands, current_binning=self.current_binning, target_binning=target_binning)
            self.current_binning = target_binning
        if self.rgb is not None:
            arr = bin_cube(np.array(self.rgb), y_factor=y, x_factor=x, lambda_factor=1)
            self.rgb = Image.fromarray(arr.astype(np.uint8))
        return self

    def get_rgb(self, which_data='raw', color_bands=None):
        """Create an RGB preview from selected spectral bands."""
        data = getattr(self, which_data)
        if color_bands is not None:
            self.color_bands = color_bands
        self.rgb = get_rgb(data, **self.color_bands)
        return self

    def view(self, ax=None):
        """Display the RGB preview on a Matplotlib axes."""
        if ax is not None:
            self.ax = ax
        else:
            _, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.imshow(self.rgb)
        return self

    def point(self, x, y, color='red', marker='+', ax=None):
        """Mark and store a pixel coordinate; return this cube."""
        ax = ax if ax is not None else self.ax
        ax.scatter(x, y, color=color, marker=marker)
        self.point_x = x
        self.point_y = y
        return self

    def vline(self, x, color='red', ax=None, **kwargs):
        """Draw and store a vertical line; return this cube."""
        ax = ax if ax is not None else self.ax
        ax.axvline(x, color=color, **kwargs)
        self.vline_x = x
        return self

    def hline(self, y, color='red', ax=None, **kwargs):
        """Draw and store a horizontal line; return this cube."""
        ax = ax if ax is not None else self.ax
        ax.axhline(y, color=color, **kwargs)
        self.hline_y = y
        return self

    def roi(self, x, y, width, height, edgecolor='red', facecolor='none', ax=None):
        """Draw and store a rectangular region; return this cube."""
        ax = ax if ax is not None else self.ax
        rect = plt.Rectangle((x, y), width, height, edgecolor=edgecolor, facecolor=facecolor)
        ax.add_patch(rect)
        self.roi_x = x
        self.roi_y = y
        self.roi_width = width
        self.roi_height = height
        return self

    def __repr__(self):
        """Return a short cube description."""
        return f'<Cube: {self.name}>'

def read_cube_envi(data_path, format='raw'):
    """Load a capture-directory ENVI cube as a float32 array."""
    header_file = list(Path(data_path, 'capture').glob('*.hdr'))[0]
    data_file = list(Path(data_path, 'capture').glob(f'*.{format}'))[0]
    cube = spy.envi.open(header_file, data_file).load()
    cube = np.array(cube, dtype=np.float32)
    return cube

def read_wvls_envi(data_path):
    """Read capture-directory ENVI wavelengths as floats."""
    header_file = list(Path(data_path, 'capture').glob('*.hdr'))[0]
    wvls = spy.envi.read_envi_header(header_file)['wavelength']
    wvls = [float(wvl) for wvl in wvls]
    return wvls

def read_fwhm_envi(data_path):
    """Read capture-directory ENVI band widths as floats."""
    header_file = list(Path(data_path, 'capture').glob('*.hdr'))[0]
    fwhm = spy.envi.read_envi_header(header_file)['fwhm']
    fwhm = [float(f) for f in fwhm]
    return fwhm
