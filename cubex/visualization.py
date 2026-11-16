"""RGB previews of hyperspectral data."""

import numpy as np

__all__ = ['get_rgb']

def get_rgb(cube, blue=None, green=None, red=None):
    """Create an RGB preview from selected spectral bands."""
    from PIL import Image
    channels = []
    for bands in (red, green, blue):
        if bands is None:
            selected = cube
        elif isinstance(bands, (int, np.integer)):
            selected = cube[:, :, [bands]]
        else:
            selected = cube[:, :, bands]
        channel = np.mean(selected, axis=-1)
        span = np.max(channel) - np.min(channel)
        channels.append((channel - np.min(channel)) / span if span else np.zeros_like(channel))
    normalized_red, normalized_green, normalized_blue = channels
    rgb_array = np.stack([normalized_red, normalized_green, normalized_blue], axis=-1)
    rgb_image = Image.fromarray((rgb_array * 255).astype(np.uint8))
    return rgb_image
