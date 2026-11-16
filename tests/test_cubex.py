"""Regression checks using small synthetic imaging data."""

import ast
import importlib
from pathlib import Path

import numpy as np
import pytest
import spectral as spy

from cubex.cubes import Cube, read_cube_envi, read_wvls_envi, read_fwhm_envi
from cubex.preprocessing import bin_cube, bin_wavelengths, bin_bands, bin_coord, unbin_coord
from cubex.spectra import FWHM
from cubex.thermography import LinRegCal, TMCal, ThermalVideo, frame_to_time, frame_to_second, second_to_frame
from cubex.visualization import get_rgb


def test_public_api_and_docstrings():
    for name in ('cubes', 'preprocessing', 'visualization', 'spectra', 'thermography'):
        module = importlib.import_module(f'cubex.{name}')
        for symbol in module.__all__:
            assert callable(getattr(module, symbol))
        for node in ast.walk(ast.parse(Path(module.__file__).read_text(encoding='utf-8'))):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                assert ast.get_docstring(node), node.name


def test_binning():
    data = np.arange(64).reshape(4, 4, 4)
    result = bin_cube(data, 2, 2, 2)
    expected = np.array([[[data[y:y+2, x:x+2, b:b+2].mean()
                           for b in (0, 2)] for x in (0, 2)] for y in (0, 2)])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(bin_wavelengths([400, 410, 420, 430], 2), [405, 425])
    assert bin_coord(8, 2, 4) == 4
    assert unbin_coord(4, 4, 2) == (8, 10)
    assert bin_bands(dict(blue=(4, 8), green=(8, 12), red=(12, 16)), 1, 2)['red'] == (6, 8)
    for factor in (0, -1, 1.5, 3):
        with pytest.raises(ValueError):
            bin_cube(data, lambda_factor=factor)


def test_envi_cube_workflow(tmp_path):
    capture = tmp_path / 'capture'
    capture.mkdir()
    data = np.arange(4 * 4 * 200, dtype=np.float32).reshape(4, 4, 200)
    wavelengths = np.arange(200) + 400
    spy.envi.save_image(str(capture / 'sample.hdr'), data, ext='.raw',
                       metadata={'wavelength': wavelengths.tolist(), 'fwhm': [2.] * 200})
    np.testing.assert_allclose(read_cube_envi(tmp_path), data)
    np.testing.assert_allclose(read_wvls_envi(tmp_path), wavelengths)
    np.testing.assert_allclose(read_fwhm_envi(tmp_path), 2.)
    cube = Cube(str(tmp_path))
    assert cube.processed is None and cube.rgb is None
    cube.bin(wvl=2).bin(wvl=2)
    assert cube.current_binning == 4
    assert cube.raw.shape == (4, 4, 50)
    assert len(cube.wvls) == len(cube.fwhm) == 50
    assert cube.color_bands['red'] == (45, 49)
    cube.processed = cube.raw.copy()
    assert cube.get_rgb('processed').bin(y=2, x=2) is cube
    assert cube.rgb.size == (2, 2)
    np.testing.assert_allclose(cube.raw, cube.processed)
    import matplotlib.pyplot as plt
    cube.view().point(0, 0).vline(0).hline(0).roi(0, 0, 1, 1)
    assert len(cube.ax.patches) == 1
    plt.close(cube.ax.figure)


def test_rgb_constant_channels_and_scalar_selectors():
    data = np.ones((3, 4, 5))
    with np.errstate(all='raise'):
        image = get_rgb(data, blue=0, green=slice(1, 3), red=(3, 4))
    assert image.size == (4, 3)
    assert np.asarray(image).max() == 0


def test_peak_interpolates_both_crossings():
    result = FWHM([0, 1, 2, 3, 4], [0, 2, 6, 2, 0])
    assert result.x_left == pytest.approx(1.25)
    assert result.x_right == pytest.approx(2.75)
    assert result.fwhm == pytest.approx(1.5)
    with pytest.raises(ValueError, match='both sides'):
        FWHM([0, 1, 2], [3, 6, 3])


def test_calibrations_and_timing():
    linear = LinRegCal([[0, 10], [1, 12], [2, 14]])
    assert linear.get_y(3) == pytest.approx(16)
    data = np.array([[[0., 1.], [2., 3.], [4., 5.]]])
    cal = TMCal(data, [(0, 0, 1, 1), (1, 0, 1, 1)], [10., 14.])
    np.testing.assert_allclose(cal.get_temp((2, 0, 1, 1)), [18, 18])
    assert frame_to_time(125, 2) == (1, 2)
    assert frame_to_second(0, 2) == 0.5
    assert second_to_frame(0.6, 2) == 2


def test_thermal_video_sampling(tmp_path):
    av = pytest.importorskip('av')
    path = tmp_path / 'thermal.mp4'
    with av.open(str(path), mode='w') as container:
        stream = container.add_stream('mpeg4', rate=10)
        stream.width = stream.height = 16
        stream.pix_fmt = 'yuv420p'
        for value in (20, 40, 60, 80):
            frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), value, dtype=np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    video = ThermalVideo(str(path))
    try:
        assert video.frames == 4
        video.get_data(2)
        assert video.data.shape == (16, 16, 2)
        assert video.data[:, :, 0].mean() == pytest.approx(30, abs=4)
        assert video.data[:, :, 1].mean() == pytest.approx(70, abs=4)
        with pytest.raises(ValueError):
            video.get_data(5)
    finally:
        video.close()
