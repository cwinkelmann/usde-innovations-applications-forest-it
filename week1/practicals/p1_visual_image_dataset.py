import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _csv_inspect(mo):
    mo.md(r"""
    # Practical 1 — Visual Wildlife Datasets

    **Context:** Wildlife data comes from very different sources with very
    different annotation formats. Before building any model, you need to
    understand what the data looks like.

    | Source | Annotation types |
    |--------|-----------------|
    | **Camera traps** (ground-level) | Image classification, bounding boxes |
    | **Aerial / drone imagery** | Bounding boxes, point annotations |

    We also preview **segmentation masks** — the pixel-level format from Practical 7.

    By the end you will:
    - Recognise how camera trap images differ from aerial tiles
    - Understand four annotation formats: image labels, bounding boxes, points, masks
    - Know which format each tool uses (MegaDetector → boxes, HerdNet → points, SAM → masks)
    """)
    return


@app.cell
def _imports():
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image

    return Image, np, pd, plt


@app.cell
def _format_detail():
    """Set up download module path."""
    import sys as _sys
    from pathlib import Path as _Path

    DATA_BASE = _Path(__file__).parent.parent / "data"
    if str(DATA_BASE) not in _sys.path:
        _sys.path.insert(0, str(DATA_BASE))
    return (DATA_BASE,)


@app.cell
def _load(mo):
    mo.md(r"""
    ---

    ## Camera Trapping

    Camera traps are motion-triggered cameras at fixed locations — ground-level,
    often at night, typically one animal per frame.

    ### Image Classification

    The simplest annotation: **one label per image**, no spatial information.
    This is what camera trap platforms (Wildlife Insights, Agouti) and
    MegaDetector's downstream classifiers output.
    """)
    return


@app.cell
def _(DATA_BASE):
    """Download camera trap datasets (Serengeti + Caltech)."""
    from download_data import download_serengeti, download_caltech
    download_serengeti(n_images=50, output_dir=DATA_BASE)
    download_caltech(n_images=50, output_dir=DATA_BASE)
    return


@app.cell
def _tile_dist(DATA_BASE, Image, np, pd, plt):
    """Load camera trap data and show species distributions + sample images."""
    import json as _json

    # ── Load metadata ─────────────────────────────────────────────────────────
    _meta_path = DATA_BASE / "camera_trap" / "serengeti_meta.json"
    serengeti_meta = None
    if _meta_path.exists():
        with open(_meta_path) as _f:
            serengeti_meta = _json.load(_f)
        _cat_map = {c["id"]: c["name"] for c in serengeti_meta["categories"]}
        print(f"Serengeti : {len(serengeti_meta['images'])} images")

    _caltech_path = DATA_BASE / "camera_trap_labels.csv"
    caltech_df = pd.read_csv(_caltech_path) if _caltech_path.exists() else None
    if caltech_df is not None:
        print(f"Caltech   : {len(caltech_df)} images, {caltech_df['true_label'].nunique()} species")

    # ── Species distribution ──────────────────────────────────────────────────
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 4))

    if serengeti_meta is not None:
        _cat_map = {c["id"]: c["name"] for c in serengeti_meta["categories"]}
        _labels = [_cat_map.get(a["category_id"], "unknown")
                   for a in serengeti_meta["annotations"]]
        _counts = pd.Series(_labels).value_counts()
        _axes[0].barh(_counts.index, _counts.values, color="steelblue")
        _axes[0].set_xlabel("Annotations")
        _axes[0].set_title(f"Snapshot Serengeti — {len(_counts)} classes")

    if caltech_df is not None:
        _counts2 = caltech_df["true_label"].value_counts()
        _axes[1].barh(_counts2.index, _counts2.values, color="darkorange")
        _axes[1].set_xlabel("Images")
        _axes[1].set_title(f"Caltech — {len(_counts2)} classes")

    plt.tight_layout()

    # ── Sample images ─────────────────────────────────────────────────────────
    _n_each = 3
    _fig2, _axes2 = plt.subplots(2, _n_each, figsize=(_n_each * 4, 8))

    if serengeti_meta is not None:
        _img_label = {}
        for _a in serengeti_meta["annotations"]:
            _img_label[_a["image_id"]] = _cat_map.get(_a["category_id"], "unknown")
        _shown = 0
        for _img_data in serengeti_meta["images"]:
            if _shown >= _n_each:
                break
            _label = _img_label.get(_img_data["id"], "empty")
            if _label == "empty":
                continue
            _path = DATA_BASE / "camera_trap" / "serengeti_subset" / _img_data["file_name"].split("/")[-1]
            if not _path.exists():
                continue
            _axes2[0][_shown].imshow(np.array(Image.open(_path)))
            _axes2[0][_shown].set_title(f"Serengeti — {_label}", fontsize=9)
            _axes2[0][_shown].axis("off")
            _shown += 1

    if caltech_df is not None:
        _caltech_dir = DATA_BASE / "camera_trap" / "caltech_subset"
        _shown = 0
        for _, _row in caltech_df.iterrows():
            if _shown >= _n_each:
                break
            _path = _caltech_dir / _row["crop"]
            if not _path.exists():
                continue
            _axes2[1][_shown].imshow(np.array(Image.open(_path)))
            _axes2[1][_shown].set_title(f"Caltech — {_row['true_label']}", fontsize=9)
            _axes2[1][_shown].axis("off")
            _shown += 1

    plt.suptitle("Camera trap examples — image-level species labels", fontsize=12)
    plt.tight_layout()
    plt.gca()
    return (caltech_df,)


@app.cell
def _tile_inspect(mo):
    mo.md(r"""
    ### Bounding Boxes

    A bounding box marks **where** an animal is. Caltech includes ground-truth
    boxes in COCO format: `[x, y, w, h]` (top-left corner + size, pixels).

    | Format | Coordinates | Used by |
    |--------|-------------|---------|
    | **COCO** | `[x, y, w, h]` — top-left + size, pixels | COCO JSON, MegaDetector |
    | **YOLO** | `[cx, cy, w, h]` — centre + size, normalised 0-1 | Ultralytics, training |
    | **Pascal VOC** | `[x_min, y_min, x_max, y_max]` — corners, pixels | VOC XML |
    """)
    return


@app.cell
def _(DATA_BASE, Image, caltech_df, np, plt):
    """Caltech images with ground-truth bounding boxes drawn."""
    import matplotlib.patches as _patches

    if caltech_df is not None:
        _with_box = caltech_df.dropna(subset=["bbox_x"])
        _n_show = min(6, len(_with_box))
        _ncols = 3
        _nrows = (_n_show + _ncols - 1) // _ncols
        _caltech_dir = DATA_BASE / "camera_trap" / "caltech_subset"

        _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(_ncols * 4, _nrows * 4))
        _axes_flat = np.array(_axes).flatten()

        _shown = 0
        for _, _row in _with_box.iterrows():
            if _shown >= _n_show:
                break
            _path = _caltech_dir / _row["crop"]
            if not _path.exists():
                continue
            _img = np.array(Image.open(_path))
            _ax = _axes_flat[_shown]
            _ax.imshow(_img)
            _rect = _patches.Rectangle(
                (_row["bbox_x"], _row["bbox_y"]),
                _row["bbox_w"], _row["bbox_h"],
                linewidth=2, edgecolor="lime", facecolor="none",
            )
            _ax.add_patch(_rect)
            _ax.set_title(f"{_row['true_label']}\n"
                          f"[{int(_row['bbox_x'])}, {int(_row['bbox_y'])}, "
                          f"{int(_row['bbox_w'])}, {int(_row['bbox_h'])}]", fontsize=8)
            _ax.axis("off")
            _shown += 1

        for _ax in _axes_flat[_shown:]:
            _ax.axis("off")

        plt.suptitle(f"Ground-truth bounding boxes — COCO format  "
                     f"({len(_with_box)}/{len(caltech_df)} have boxes)", fontsize=11)
        plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Aerial Datasets

    Drone imagery: camera points **straight down**, animals are **tiny**
    (often < 50 pixels), and most tiles are empty.

    | Dataset | Annotation type |
    |---------|-----------------|
    | **Eikelboom 2019** | Bounding boxes (YOLO `.txt` format) |
    | **HerdNet General Dataset** | Point annotations (x, y per animal) |

    **When do you need a box, and when is a point enough?**

    ### Bounding Boxes — Eikelboom 2019
    """)
    return


@app.cell
def _species_gallery(DATA_BASE):
    """Download Eikelboom aerial dataset."""
    from download_data import download_eikelboom
    download_eikelboom(n_images=50, output_dir=DATA_BASE)
    return


@app.cell
def _classes(DATA_BASE, Image, np, plt):
    """Show Eikelboom aerial samples and YOLO label statistics."""
    _eik_dir = DATA_BASE / "eikelboom"
    _train_dir = _eik_dir / "train"

    for _split in ["train", "val", "test"]:
        _split_dir = _eik_dir / _split
        if _split_dir.exists():
            _n = len([p for p in _split_dir.iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            print(f"  {_split}: {_n} images")

    # Show train samples
    if _train_dir.exists():
        _imgs = sorted(
            p for p in _train_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )[:6]

        if _imgs:
            _ncols = min(3, len(_imgs))
            _nrows = (len(_imgs) + _ncols - 1) // _ncols
            _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(_ncols * 4, _nrows * 4))
            _axes_flat = np.array(_axes).flatten() if len(_imgs) > 1 else [_axes]

            for _i, _path in enumerate(_imgs):
                _axes_flat[_i].imshow(np.array(Image.open(_path)))
                _axes_flat[_i].set_title(_path.name, fontsize=8)
                _axes_flat[_i].axis("off")
            for _ax in _axes_flat[len(_imgs):]:
                _ax.axis("off")

            plt.suptitle("Eikelboom 2019 — aerial wildlife tiles (train split)", fontsize=11)
            plt.tight_layout()

    # Parse YOLO labels
    _label_files = sorted(_train_dir.glob("*.txt")) if _train_dir.exists() else []
    _class_counts = {}
    for _lf in _label_files:
        for _line in _lf.read_text().strip().splitlines():
            _parts = _line.strip().split()
            if len(_parts) >= 5:
                _class_counts[_parts[0]] = _class_counts.get(_parts[0], 0) + 1

    if _class_counts:
        import pandas as _pd
        _counts = _pd.Series(_class_counts).sort_values(ascending=False)
        _fig2, _ax = plt.subplots(figsize=(6, 2.5))
        _ax.barh([f"class {c}" for c in _counts.index], _counts.values, color="teal")
        _ax.set_xlabel("Bounding boxes")
        _ax.set_title("Eikelboom — boxes per class (train)")
        plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Point Annotations — HerdNet General Dataset

    The **lightest** spatial annotation: one `(x, y)` per animal, no box needed.
    Reduces annotation effort dramatically for dense colonies.

    | Column | Meaning |
    |--------|---------|
    | `images` | Tile filename |
    | `x`, `y` | Pixel coordinate within the tile |
    | `labels` | Species class |

    **One row = one animal.** Empty tiles have no rows.
    This is the format used by **HerdNet** and the **Iguanas From Above** project.
    """)
    return


@app.cell
def _(DATA_BASE):
    """Download HerdNet General Dataset."""
    from download_data import download_general_dataset
    download_general_dataset(n_images=50, output_dir=DATA_BASE)
    return


@app.cell
def _gallery(DATA_BASE, Image, np, pd, plt):
    """Load HerdNet data: CSV summary, class distribution, top-3 tiles."""
    DATA_DIR = DATA_BASE / "general_dataset" / "test_sample"
    CSV_PATH = DATA_BASE / "general_dataset" / "test_sample.csv"

    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        tile_files = sorted(
            p for p in DATA_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}
        )

        print(f"Tiles: {len(tile_files)}  |  Annotations: {len(df)}  |  "
              f"Annotated tiles: {df['images'].nunique()}")
        print(f"\nFirst 5 rows:\n{df.head().to_string(index=False)}")

        # ── Class distribution ────────────────────────────────────────────────
        _counts = df["labels"].value_counts()
        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 3.5))

        _ax1.barh(_counts.index, _counts.values, color="steelblue")
        _ax1.set_xlabel("Annotations")
        _ax1.set_title("Annotations per species")

        _per_tile = df.groupby("images").size()
        _ax2.hist(_per_tile.values, bins=20, color="steelblue", edgecolor="white")
        _ax2.set_xlabel("Annotations per tile")
        _ax2.set_ylabel("Tiles")
        _ax2.set_title(f"Distribution (median={_per_tile.median():.0f}, max={_per_tile.max()})")
        plt.tight_layout()

        # ── Top-3 densest tiles with annotations ──────────────────────────────
        _top3 = _per_tile.sort_values(ascending=False).head(3).index.tolist()
        _fig2, _axes = plt.subplots(1, 3, figsize=(15, 5))
        for _ax, _name in zip(_axes, _top3):
            _img = np.array(Image.open(DATA_DIR / _name))
            _tile_df = df[df["images"] == _name]
            _ax.imshow(_img)
            _ax.scatter(_tile_df["x"], _tile_df["y"],
                        c="red", s=25, marker="+", linewidths=1.5)
            _ax.set_title(f"{_name}\n({len(_tile_df)} animals)", fontsize=9)
            _ax.axis("off")
        plt.suptitle("Three most densely annotated tiles", fontsize=12)
        plt.tight_layout()
    else:
        print("CSV not found — check download output above.")
    plt.gca()
    return


@app.cell
def _step6(mo):
    mo.md(r"""
    ---

    ## Segmentation Masks (Preview)

    The fourth annotation type: every pixel gets a class label. Most detailed,
    most expensive. You will work with this in **Practical 7** (SAM).

    ## Summary — Annotation Types at a Glance

    | Type | Spatial info | Effort | Used by | Best for |
    |------|-------------|--------|---------|----------|
    | **Image label** | None | Lowest | DeepFaune, SpeciesNet | Species ID after detection |
    | **Bounding box** | Rectangle | Medium | MegaDetector, YOLOv8 | Object detection, counting |
    | **Point** | Centre (x,y) | Low | HerdNet | Dense colonies, aerial |
    | **Segmentation mask** | Pixel outline | Highest | SAM, U-Net | Habitat mapping, area estimation |

    The right choice depends on your ecological question — more detail is not
    always better when annotation cost is the bottleneck.
    """)
    return


@app.cell
def _step2(mo):
    mo.md(r"""
    ## Exercise

    1. Compare a camera trap image with an aerial tile. List three visual
       differences that would affect a detection model.
    2. The Caltech dataset has bounding boxes; HerdNet has points.
       Which would you choose for counting iguanas on a rocky beach? Why?
    3. Look at the Eikelboom aerial images. How do they compare in resolution
       and animal size to the HerdNet tiles?
    4. The point annotations are pixel coordinates. What extra information
       would you need to convert them to GPS positions?
    """)
    return


@app.cell
def _step3(mo):
    mo.md(r"""
    ## Reflection

    - Camera traps and drones produce very different data. How should this
      affect your choice of detection model?
    - Empty tiles have no rows in the CSV. What does this mean for training
      a model that must also learn to say "nothing here"?
    - The four annotation types form a hierarchy of increasing detail. Is
      more detail always better, or is there a cost?
    """)
    return


if __name__ == "__main__":
    app.run()
