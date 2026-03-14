"""Tile a plain image and export bounding box annotations in YOLO format."""

from pathlib import Path

import numpy as np
import pandas as pd

from wildlife_detection.tiling.utils import (
    generate_tile_windows,
    load_image_array,
    read_tile,
    save_tile_jpeg,
)


def tile_boxes(image_path, annotations_path, tile_size=512, overlap=120,
               output_dir="data/processed", min_visibility=0.5):
    """Tile an image and convert bounding boxes to per-tile YOLO label files.

    Accepts annotations in two formats:

    **YOLO txt** (``annotations_path`` is a ``.txt`` file):
        One row per box: ``class_id cx cy w h`` normalised to [0, 1].

    **CSV** (``annotations_path`` is a ``.csv`` file):
        Columns: ``class_id``, ``x_min``, ``y_min``, ``x_max``, ``y_max``
        in absolute pixel coordinates.

    Boxes whose intersection with a tile covers less than ``min_visibility``
    of the original box area are discarded.

    Output YOLO label format per line: ``class_id cx cy w h`` normalised
    to [0, 1] relative to ``tile_size``.

    Parameters
    ----------
    image_path : str or Path
        Path to source image.
    annotations_path : str or Path
        Path to annotation file (.txt YOLO or .csv).
    tile_size : int
        Tile size in pixels (default 512).
    overlap : int
        Overlap between adjacent tiles in pixels (default 120).
    output_dir : str or Path
        Destination directory. Creates ``images/`` and ``labels/``.
    min_visibility : float
        Minimum fraction of box area that must remain in a tile (default 0.5).

    Returns
    -------
    pathlib.Path
        Path to the output directory.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_array = load_image_array(image_path)
    image_height, image_width = image_array.shape[:2]

    # Load annotations into absolute pixel coordinates
    annotations_path = Path(annotations_path)
    if annotations_path.suffix == ".txt":
        boxes = _load_yolo_txt(annotations_path, image_width, image_height)
    else:
        boxes = _load_csv(annotations_path)

    stem = Path(image_path).stem
    n_boxes_total = 0

    for window in generate_tile_windows(image_width, image_height, tile_size, overlap):
        col_off, row_off, win_width, win_height = window

        tile_name = f"{stem}_{col_off}_{row_off}"
        tile_array = read_tile(image_array, window, tile_size)
        save_tile_jpeg(tile_array, images_dir / f"{tile_name}.jpg")

        lines = []
        for class_id, x1, y1, x2, y2 in boxes:
            # Intersect box with tile
            ix1 = max(x1, col_off)
            iy1 = max(y1, row_off)
            ix2 = min(x2, col_off + win_width)
            iy2 = min(y2, row_off + win_height)

            if ix2 <= ix1 or iy2 <= iy1:
                continue

            orig_area = (x2 - x1) * (y2 - y1)
            inter_area = (ix2 - ix1) * (iy2 - iy1)
            if orig_area > 0 and inter_area / orig_area < min_visibility:
                continue

            # Convert to local tile coords, normalise to [0, 1]
            lx1 = ix1 - col_off
            ly1 = iy1 - row_off
            lx2 = ix2 - col_off
            ly2 = iy2 - row_off

            cx = (lx1 + lx2) / 2 / tile_size
            cy = (ly1 + ly2) / 2 / tile_size
            w  = (lx2 - lx1) / tile_size
            h  = (ly2 - ly1) / tile_size

            lines.append(f"{int(class_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            n_boxes_total += 1

        (labels_dir / f"{tile_name}.txt").write_text("\n".join(lines))

    n_tiles = len(list(images_dir.glob("*.jpg")))
    print(f"YOLO dataset created: {n_tiles} tiles, {n_boxes_total} boxes → {output_dir}")
    return output_dir


def _load_yolo_txt(path, image_width, image_height):
    """Parse a YOLO txt file into absolute pixel coordinates."""
    boxes = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        class_id = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = (cx - w / 2) * image_width
        y1 = (cy - h / 2) * image_height
        x2 = (cx + w / 2) * image_width
        y2 = (cy + h / 2) * image_height
        boxes.append((class_id, x1, y1, x2, y2))
    return boxes


def _load_csv(path):
    """Load a CSV with columns class_id, x_min, y_min, x_max, y_max."""
    df = pd.read_csv(path)
    return list(zip(
        df["class_id"].astype(int),
        df["x_min"].astype(float),
        df["y_min"].astype(float),
        df["x_max"].astype(float),
        df["y_max"].astype(float),
    ))
