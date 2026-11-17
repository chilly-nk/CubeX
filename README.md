# CubeX

Reusable hyperspectral and thermal imaging tools extracted from the **Functions (module)** section of `CubeX_v0.5   261020_PFA_RatHeart_HSI_IRT_v1.ipynb`. The original notebook remains unchanged; dataset-specific analysis, Colab setup, and drafts stay there.

## Installation

From this repository, with Python 3.10 or newer:

```sh
python -m pip install .
```

For thermal video decoding, install `".[video]"`. For development and tests, install `-e ".[dev,video]"`.

## Usage

```python
from cubex.cubes import Cube
from cubex.preprocessing import bin_cube, bin_wavelengths
from cubex.visualization import get_rgb
from cubex.spectra import FWHM
from cubex.thermography import TMCal, LinRegCal, ThermalVideo

cube = Cube("path/to/specim_acquisition")
cube.processed = cube.raw - dark_reference  # Supply your own dark-reference array.
cube.get_rgb(which_data="processed").view().roi(10, 20, 30, 40)

peak = FWHM(cube.wvls, cube.processed[20, 10, :])
print(peak.fwhm)
```

`Cube` expects `capture/*.hdr` and `capture/*.raw` under the acquisition directory. The header supplies `wavelength` and `fwhm`. Arrays use `(y, x, wavelength)` for hyperspectral cubes and `(y, x, frame)` for thermal data. `processed` starts as `None`; assign your corrected data before using it.

## API

| Module | Public classes and functions |
| --- | --- |
| `cubex.cubes` | `Cube`, `read_cube_envi`, `read_wvls_envi`, `read_fwhm_envi` |
| `cubex.preprocessing` | `bin_cube`, `bin_wavelengths`, `unbin_coord`, `bin_coord`, `bin_bands` |
| `cubex.visualization` | `get_rgb` |
| `cubex.spectra` | `FWHM` |
| `cubex.thermography` | `TMCal`, `LinRegCal`, `ThermalVideo`, `frame_to_time`, `frame_to_second`, `second_to_frame` |

`Cube` provides chainable `bin`, `get_rgb`, `view`, `point`, `vline`, `hline`, and `roi` methods. Call `view` before annotation methods unless you supply an axes.

## Conventions and extraction fixes

- Binning averages samples and requires divisible dimensions. `Cube.bin` accepts factors relative to the current cube and tracks cumulative spectral binning. Wavelengths and supplied FWHM metadata are averaged per bin; averaged FWHM metadata is not a model of the effective binned spectral response.
- RGB channel selectors retain NumPy indexing: `(33, 49)` selects two bands, while `slice(33, 49)` selects a range. Each channel is independently normalized; constant channels become black. Default Specim selectors require at least 200 bands; supply selectors for other sensors.
- `FWHM` expects finite samples with increasing wavelengths and one peak crossing half maximum on both sides. The descending-side interpolation is corrected from the notebook.
- `TMCal` fits each frame independently using reference ROIs `(x, y, width, height)` and known temperatures. `LinRegCal` accepts rows of `(x, y)` reference pairs.
- `ThermalVideo` preserves the notebook's red-channel extraction from decoded `rgb24` images; this is not raw radiometric temperature decoding. It requires constant frame timing and at least two initial frames. Call `close()` when finished.
- Timing helpers retain the notebook conventions: `frame_to_time(0, fps)` is `(0, 0)`; `frame_to_second(0, fps)` is `1 / fps` (end of exposure); `second_to_frame` returns `ceil(seconds * fps)`. They are not inverse index conversions.
- Imports are explicit; `Cube` accepts string paths, initializes optional state, and rejects unsupported origins.

## Development

### Version tracking

The current package version is `0.6.0`. See [CHANGELOG.md](CHANGELOG.md) for release
history and changes planned for the next release.

Commit meaningful changes as you work. Before a release, update the version in
`pyproject.toml`, move the relevant changelog entries from `Unreleased` into a dated
release section, and run the tests and package build below. Commit those changes,
then create and push an annotated tag matching the version (for example, `v0.5.0`).
Use patch versions for bug fixes and minor versions for new features; document any
breaking changes explicitly while the package is below version 1.0. Reserve 1.0.0
for the first stable public API.

```sh
python -m pytest
python -m build
```

GitHub Actions runs tests and package builds on Python 3.10 and 3.12. Choose a license before distributing this code; no license or authorship has been assumed.
