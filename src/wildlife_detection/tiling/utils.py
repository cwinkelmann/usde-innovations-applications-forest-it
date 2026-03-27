"""Shared tiling utilities for sliding-window image tiling."""

from collections import namedtuple
from pathlib import Path

import numpy as np
from PIL import Image

# Lightweight replacement for rasterio.windows.Window
TileWindow = namedtuple("TileWindow", ["col_off", "row_off", "width", "height"])


def generate_tile_windows(image_width, image_height, tile_size, overlap):
    """Generate tile windows covering the full image extent.

    Parameters
    ----------
    image_width : int
    image_height : int
    tile_size : int
        Square tile size in pixels.
    overlap : int
        Pixel overlap between adjacent tiles.

    Yields
    ------
    TileWindow
    """
    step = tile_size - overlap
    for row_off in range(0, image_height, step):
        for col_off in range(0, image_width, step):
            win_width = min(tile_size, image_width - col_off)
            win_height = min(tile_size, image_height - row_off)
            yield TileWindow(col_off, row_off, win_width, win_height)


def read_tile(image_array, window, tile_size):
    """Crop a tile from an (H, W, 3) uint8 numpy array, padding edge tiles.

    Parameters
    ----------
    image_array : numpy.ndarray
        RGB image array of shape (H, W, 3), dtype uint8.
    window : TileWindow
    tile_size : int
        Output tile size (used for padding).

    Returns
    -------
    numpy.ndarray
        Shape (tile_size, tile_size, 3), dtype uint8.
    """
    col_off, row_off, win_width, win_height = window
    crop = image_array[row_off:row_off + win_height, col_off:col_off + win_width]

    if win_height < tile_size or win_width < tile_size:
        padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        padded[:win_height, :win_width] = crop
        return padded

    return crop.copy()


def save_tile_jpeg(array, path, quality=95):
    """Save a numpy RGB array as JPEG.

    Parameters
    ----------
    array : numpy.ndarray
        Shape (H, W, 3), dtype uint8.
    path : str or Path
    quality : int
    """
    Image.fromarray(array).save(path, quality=quality)


def find_class_column(df):
    """Find the class label column in a DataFrame or GeoDataFrame.

    Checks common names in priority order: class_label, class, label,
    species, category.

    Returns
    -------
    str or None
    """
    for candidate in ["class_label", "class", "label", "species", "category"]:
        if candidate in df.columns:
            return candidate
    return None


def load_image_array(image_path):
    """Load an image file as an (H, W, 3) uint8 numpy array.

    Parameters
    ----------
    image_path : str or Path

    Returns
    -------
    numpy.ndarray
        Shape (H, W, 3), dtype uint8.
    """
    img = Image.open(image_path).convert("RGB")
    return np.array(img, dtype=np.uint8)
