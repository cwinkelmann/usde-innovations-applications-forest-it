"""Density map generation for point-based detection models (HerdNet / FIDT)."""

import numpy as np
from scipy.ndimage import distance_transform_edt


def generate_fidt_map(points, height, width, radius=1):
    """Generate a Focal Inverse Distance Transform density map from point annotations.

    Each point produces a sharp peak that decays with inverse distance,
    maintaining distinct peaks even in dense regions (unlike Gaussian kernels).

    Parameters
    ----------
    points : list of (x, y) tuples
        Point coordinates in pixel space (col, row).
    height : int
        Output map height in pixels.
    width : int
        Output map width in pixels.
    radius : int
        Radius of the foreground marker around each point.

    Returns
    -------
    numpy.ndarray
        Float32 array of shape (height, width) with values in [0, 1].
    """
    if len(points) == 0:
        return np.zeros((height, width), dtype=np.float32)

    binary = np.zeros((height, width), dtype=np.uint8)
    for x, y in points:
        col = int(round(x))
        row = int(round(y))
        if 0 <= row < height and 0 <= col < width:
            y_min, y_max = max(0, row - radius), min(height, row + radius + 1)
            x_min, x_max = max(0, col - radius), min(width, col + radius + 1)
            binary[y_min:y_max, x_min:x_max] = 1

    if binary.sum() == 0:
        return np.zeros((height, width), dtype=np.float32)

    dist = distance_transform_edt(1 - binary)
    fidt = 1.0 / (1.0 + dist)

    return fidt.astype(np.float32)
