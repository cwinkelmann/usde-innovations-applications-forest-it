"""Tile a plain image and its paired segmentation mask."""

from pathlib import Path

import numpy as np
from PIL import Image

from wildlife_detection.tiling.utils import (
    generate_tile_windows,
    load_image_array,
    read_tile,
    save_tile_jpeg,
)


def tile_masks(image_path, mask_path, tile_size=512, overlap=120,
               output_dir="data/processed"):
    """Tile an image and its paired single-channel segmentation mask.

    The mask must be a single-channel image (PNG) the same size as the source
    image, where each pixel value is a class index (0 = background).

    Parameters
    ----------
    image_path : str or Path
        Path to source image (JPEG, PNG, or any PIL-readable format).
    mask_path : str or Path
        Path to a single-channel uint8 PNG mask of the same spatial size.
    tile_size : int
        Tile size in pixels (default 512).
    overlap : int
        Overlap between adjacent tiles in pixels (default 120).
    output_dir : str or Path
        Destination directory. Creates ``raster_tiles/`` and ``mask_tiles/``.

    Returns
    -------
    pathlib.Path
        Path to the output directory.
    """
    output_dir = Path(output_dir)
    tiles_dir = output_dir / "raster_tiles"
    masks_dir = output_dir / "mask_tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    image_array = load_image_array(image_path)
    image_height, image_width = image_array.shape[:2]

    mask_array = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)

    stem = Path(image_path).stem
    n_tiles = 0

    for window in generate_tile_windows(image_width, image_height, tile_size, overlap):
        col_off, row_off, win_width, win_height = window

        tile_name = f"{stem}_{col_off}_{row_off}"

        # Image tile
        tile_array = read_tile(image_array, window, tile_size)
        save_tile_jpeg(tile_array, tiles_dir / f"{tile_name}.jpg")

        # Mask tile — pad with zeros (background) at edges
        mask_crop = mask_array[row_off:row_off + win_height, col_off:col_off + win_width]
        if win_height < tile_size or win_width < tile_size:
            mask_tile = np.zeros((tile_size, tile_size), dtype=np.uint8)
            mask_tile[:win_height, :win_width] = mask_crop
        else:
            mask_tile = mask_crop.copy()

        Image.fromarray(mask_tile).save(masks_dir / f"{tile_name}.png")
        n_tiles += 1

    print(f"Segmentation dataset created: {n_tiles} tiles → {output_dir}")
    return output_dir
