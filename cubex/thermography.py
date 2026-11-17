"""Thermal calibration, video sampling, and timing."""

import numpy as np

__all__ = ['HeatersLog', 'TMCal', 'LinRegCal', 'ThermalVideo', 'frame_to_time', 'frame_to_second', 'second_to_frame']


class HeatersLog:
    """Read heater temperatures and output states from a status log.

    Parameters
    ----------
    path : str or path-like
        UTF-8 log beginning with ``Status interval = <number> <unit>``.

    Attributes
    ----------
    heaters : list of str
        Heater names, in their logged order.
    set_temps : list of float
        Constant set temperatures in heater order.
    temps, outs : numpy.ndarray
        Temperatures and integer output states, shaped (readings, heaters).
    interval : float
        Positive reading interval from the header.
    interval_unit : str
        Interval unit as written in the header.
    timing : numpy.ndarray
        Elapsed times starting at zero, expressed in ``interval_unit``.
    data : pandas.DataFrame
        Temperatures with heater columns and elapsed time as the index.
    """

    def __init__(self, path):
        """Parse a log, rejecting malformed or inconsistent readings."""
        import re
        import pandas as pd

        number = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
        header_pattern = rf'Status\s+interval\s*=\s*({number})\s+(\S+)'
        heater_pattern = re.compile(
            rf'(H\d+)\s*\(\s*({number})\s*\)\s*T\s*=\s*({number})\s+OUT\s*=\s*([01])'
        )
        self.heaters = []
        self.set_temps = []
        temperatures, outputs = [], []
        with open(path, encoding='utf-8-sig') as log:
            header = re.fullmatch(header_pattern, log.readline().strip())
            if header is None:
                raise ValueError('Expected first line: Status interval = <number> <unit>.')
            self.interval = float(header[1])
            self.interval_unit = header[2]
            if not np.isfinite(self.interval) or self.interval <= 0:
                raise ValueError('Status interval must be finite and positive.')
            for line_number, line in enumerate(log, start=2):
                line = line.strip()
                if not line:
                    continue
                entries = [heater_pattern.fullmatch(part.strip()) for part in line.split('|')]
                if any(entry is None for entry in entries):
                    raise ValueError(f'Malformed heater reading on line {line_number}.')
                heaters = [entry[1] for entry in entries]
                set_temps = [float(entry[2]) for entry in entries]
                temps = [float(entry[3]) for entry in entries]
                if not np.all(np.isfinite(set_temps + temps)):
                    raise ValueError(f'Non-finite temperature on line {line_number}.')
                if not self.heaters:
                    if len(set(heaters)) != len(heaters):
                        raise ValueError(f'Duplicate heater names on line {line_number}.')
                    self.heaters = heaters
                    self.set_temps = set_temps
                elif heaters != self.heaters or set_temps != self.set_temps:
                    raise ValueError(f'Heaters or set temperatures changed on line {line_number}.')
                temperatures.append(temps)
                outputs.append([int(entry[4]) for entry in entries])
        if not temperatures:
            raise ValueError('Log contains no heater readings.')
        self.temps = np.array(temperatures, dtype=float)
        self.outs = np.array(outputs, dtype=int)
        self.timing = np.arange(len(self.temps)) * self.interval
        self.data = pd.DataFrame(self.temps, columns=self.heaters, index=self.timing)

class TMCal:
    """Fit a temperature calibration for each image frame."""

    def __init__(self, data, coords, temps):
        """Calibration data for mapping image pixel values to temperature.

        Parameters
        ----------
        data : numpy.ndarray
            Image or pixel-value data used for calibration.
        coords : array-like
            Pixel coordinates specifying the locations from which calibration
            values are extracted. Each coordinate is given as (x, y, w, h).
        temps : array-like
            Known reference temperatures, either shape (n_rois,) for fixed
            temperatures shared by all frames, or (n_frames, n_rois) for
            temperatures specific to each frame. Frames are the last axis
            of data.
        """
        from scipy.stats import linregress
        self.data = data
        self.coords = coords
        self.temps = np.asarray(temps, dtype=float)
        if self.temps.ndim == 1:
            if self.temps.shape != (len(coords),):
                raise ValueError('Number of coordinates must match the number of temperatures.')
            frame_temps = np.broadcast_to(self.temps, (data.shape[-1], len(coords)))
        elif self.temps.ndim == 2:
            if self.temps.shape != (data.shape[-1], len(coords)):
                raise ValueError('2D temperatures must have shape (n_frames, n_rois).')
            frame_temps = self.temps
        else:
            raise ValueError('Temperatures must be a 1D or 2D array.')
        self.ref_pixels = []
        for i, coord in enumerate(self.coords):
            x, y, w, h = coord
            pixel = self.data[y:y + h, x:x + w, :].mean(axis=(0, 1))
            self.ref_pixels.append(pixel)
        self.ref_pixels = np.array(self.ref_pixels)
        self.fits = [linregress(self.ref_pixels[:, i], frame_temps[i]) for i in range(data.shape[-1])]
        self.slopes = np.array([fit.slope for fit in self.fits])
        self.intercepts = np.array([fit.intercept for fit in self.fits])
        self.rvalues = np.array([fit.rvalue for fit in self.fits])
        self.r_squares = self.rvalues ** 2
        self.formulas = [f'$y = {slope:.2f} \\times x + {intercept:.2f}$' for slope, intercept in zip(self.slopes, self.intercepts)]

    def get_temp(self, coords):
        """Return calibrated temperatures for an (x, y, width, height) ROI."""
        x, y, w, h = coords
        pixel = self.data[y:y + h, x:x + w, :].mean(axis=(0, 1))
        temp = self.slopes * pixel + self.intercepts
        return temp

class LinRegCal:
    """Fit a linear calibration to pairs of reference values."""

    def __init__(self, cal_data):
        """Initialize the object from the supplied data."""
        from scipy.stats import linregress
        self.data = np.array(cal_data)
        self.fit = linregress(self.data[:, 0], self.data[:, 1])
        self.slope = self.fit.slope
        self.intercept = self.fit.intercept
        self.formula = f'y = {self.slope:.2f} × x + {self.intercept:.2f}'
        self.rvalue = self.fit.rvalue

    def get_y(self, x):
        """Apply the fitted linear calibration."""
        return self.slope * x + self.intercept

class ThermalVideo:
    """Read and average constant-frame-duration thermal video."""

    def __init__(self, video_path, first_frames=10):
        """Initialize the object from the supplied data."""
        import av
        if first_frames < 2:
            raise ValueError('At least two initial frames are required.')
        self.container = av.open(video_path)
        self.video = self.container.streams.video[0]
        self.meta_frames = self.video.frames
        self._first_frames = self._get_first_frames(first_frames)
        sf = self._first_frames
        if len(sf) < 2 or any(frame.pts is None for frame in sf):
            self.close()
            raise ValueError('At least two timestamped frames are required.')
        self._some_frame_durations = [sf[i + 1].pts - sf[i].pts for i in list(range(len(sf) - 1))]
        self.constant_frame_duration = all((d == self._some_frame_durations[0] for d in self._some_frame_durations))
        if not self.constant_frame_duration or self._some_frame_durations[0] <= 0 or self.video.duration is None:
            self.close()
            raise ValueError('A known duration and constant positive frame timing are required.')
        if self.constant_frame_duration:
            self.frame_duration = self._some_frame_durations[0]
        self.frames = int(self.video.duration / self.frame_duration)
        self.start_pts = sf[0].pts

    def close(self):
        """Close the video container."""
        self.container.close()

    def _get_first_frames(self, n=10):
        """Decode up to n initial frames for timing inspection."""
        first_frames = []
        for i, frame in enumerate(self.container.decode(self.video)):
            if i < n:
                first_frames.append(frame)
            else:
                break
        return first_frames

    def get_corresponding_frame(self, ref_index, ref_framecount):
        """Average thermal frames corresponding to a reference frame."""
        if not isinstance(ref_framecount, (int, np.integer)) or not 1 <= ref_framecount <= self.frames:
            raise ValueError('Reference frame count must be between one and the video frame count.')
        if not isinstance(ref_index, (int, np.integer)) or not 0 <= ref_index < ref_framecount:
            raise ValueError('Reference index is out of range.')
        fpf = self.frames / ref_framecount
        container = self.container
        video = self.video
        frame_duration = self.frame_duration
        num_frames = int(np.floor(fpf))
        start_frame = int(np.ceil(ref_index * fpf))
        start_frame_pts = self.start_pts + start_frame * frame_duration
        frames = []
        frames_pts = []
        container.seek(start_frame_pts, stream=video)
        for frame in container.decode(video):
            if frame.pts >= start_frame_pts:
                frames_pts.append(frame.pts)
                image = frame.to_ndarray(format='rgb24')[:, :, 0]
                frames.append(image)
                if len(frames) == num_frames:
                    break
        if not frames:
            raise ValueError('No video frames were decoded for the requested interval.')
        mean_frame = np.mean(frames, axis=0, dtype=np.float32)
        print(f'Start frame: {start_frame} | Frames PTSs: {frames_pts}')
        return mean_frame

    def get_data(self, frames):
        """Sample the video into a (height, width, frames) array."""
        if not isinstance(frames, (int, np.integer)) or not 1 <= frames <= self.frames:
            raise ValueError('Frame count must be between one and the video frame count.')
        video = self.video
        fpf = self.frames / frames
        data = np.empty((video.height, video.width, frames), dtype=np.float32)
        for i in range(frames):
            print(f'Getting frame {i}...')
            frame_i = self.get_corresponding_frame(i, frames)
            data[:, :, i] = frame_i
        self.data = data
        return self

def frame_to_time(frame, frame_rate):
    """Convert a zero-based frame index to whole minutes and seconds."""
    frame_time_s = frame * 1 / frame_rate
    frame_time_m = int(frame_time_s // 60)
    remaining_time_s = int(frame_time_s - frame_time_m * 60)
    return (frame_time_m, remaining_time_s)

def frame_to_second(frame, frame_rate):
    """Return the elapsed time after frame `frame` has been captured."""
    return (frame + 1) / frame_rate

def second_to_frame(second, frame_rate):
    """Return the ceiling of elapsed seconds times frame rate."""
    frame = int(np.ceil(second * frame_rate))
    return frame
