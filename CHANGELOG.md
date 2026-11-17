# Changelog

Record user-visible changes here before each release. Package versions are set in
`pyproject.toml`; each release has a matching annotated Git tag (`vX.Y.Z`).

## Unreleased

## 0.7.0 - 2026-11-17

### Added

- `TMCal` accepts per-frame reference temperatures with shape
  `(n_frames, n_rois)`, while retaining fixed 1D reference temperatures.
- Validation of temperature array dimensions and frame/ROI counts.

## 0.6.0 - 2026-11-17

### Added

- `thermography.HeatersLog` for reading heater status logs into temperature and
  output arrays, elapsed timing, and a pandas DataFrame.
- pandas as a dependency for heater log DataFrames.

## 0.5.0 - 2026-11-16

Initial tagged release of the reusable CubeX Python package.

### Added

- Hyperspectral cube loading, binning, RGB visualization, and annotations.
- Spectral peak full width at half maximum (FWHM) analysis.
- Thermal calibration, video decoding, and frame timing helpers.
- Package installation metadata, API documentation, and automated tests and builds.

### Fixed

- Descending-side interpolation for FWHM calculations extracted from the notebook.
- Explicit imports, optional cube state initialization, and input validation.

### Repository

- Dataset-specific analysis notebooks remain local and are ignored by Git.
