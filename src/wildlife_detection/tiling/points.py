"""Tile a plain image using a point annotation CSV."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from wildlife_detection.tiling.utils import (
    find_class_column,
    generate_tile_windows,
    load_image_array,
    read_tile,
    save_tile_jpeg,
)


def tile_points(image_path, csv_path, tile_size=512, overlap=120,
                output_dir="data/processed"):
    """Tile an image and project point annotations into local tile coordinates.

    The input CSV must contain at minimum columns ``x`` and ``y`` (pixel
    coordinates relative to the full image). An optional ``class_label``
    column (or any recognised variant) is forwarded to the output CSV.

    Points within ``overlap / 2`` pixels of a tile edge are included in both
    neighbouring tiles to avoid dropping annotations at boundaries.

    Parameters
    ----------
    image_path : str or Path
        Path to source image (JPEG, PNG, or any PIL-readable format).
    csv_path : str or Path
        Path to CSV with at least columns ``x``, ``y``.
    tile_size : int
        Tile size in pixels (default 512).
    overlap : int
        Overlap between adjacent tiles in pixels (default 120).
    output_dir : str or Path
        Destination directory for tiles and output CSV.

    Returns
    -------
    pathlib.Path
        Path to the generated ``annotations_points.csv``.
    """
    output_dir = Path(output_dir)
    tiles_dir = output_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    image_array = load_image_array(image_path)
    image_height, image_width = image_array.shape[:2]

    df = pd.read_csv(csv_path)
    cols = df["x"].values.astype(float)
    rows = df["y"].values.astype(float)
    class_col = find_class_column(df)

    boundary_tolerance = overlap / 2
    stem = Path(image_path).stem
    csv_rows = []

    for window in generate_tile_windows(image_width, image_height, tile_size, overlap):
        col_off = window.col_off
        row_off = window.row_off
        win_width = window.width
        win_height = window.height

        tile_name = f"{stem}_{col_off}_{row_off}.jpg"
        tile_array = read_tile(image_array, window, tile_size)
        save_tile_jpeg(tile_array, tiles_dir / tile_name)

        in_tile = (
            (cols >= col_off - boundary_tolerance)
            & (cols < col_off + win_width + boundary_tolerance)
            & (rows >= row_off - boundary_tolerance)
            & (rows < row_off + win_height + boundary_tolerance)
        )

        for idx in np.where(in_tile)[0]:
            label = df.iloc[idx][class_col] if class_col else "animal"
            csv_rows.append({
                "tile_filename": tile_name,
                "local_x": round(float(cols[idx] - col_off), 2),
                "local_y": round(float(rows[idx] - row_off), 2),
                "class_label": label,
                "source_row": int(df.index[idx]),
            })

    csv_path_out = output_dir / "annotations_points.csv"
    with open(csv_path_out, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["tile_filename", "local_x", "local_y", "class_label", "source_row"]
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    n_tiles = len(list(tiles_dir.glob("*.jpg")))
    print(f"Tiling complete: {n_tiles} tiles, {len(csv_rows)} annotations → {csv_path_out}")
    return csv_path_out
