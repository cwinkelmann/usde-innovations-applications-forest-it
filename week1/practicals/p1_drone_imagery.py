import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _context(mo):
    mo.md(r"""
    # Practical 1 — Exploring Aerial Wildlife Data

    **Context:** Before building any model, you need to understand what the data looks like.
    This practical introduces the **HerdNet General Dataset** — pre-tiled aerial images of
    African mammals with point annotations. This is the dataset format you will work with
    throughout Week 1.

    By the end you will:
    - Know what an aerial wildlife tile looks like and how big it is
    - Understand the CSV annotation format (one row per animal)
    - See the class distribution and how annotations are spread across tiles
    - Recognise the difference between dense and empty tiles

    The cell below also downloads the camera-trap datasets used in Practicals 3–6.
    Run it once; subsequent runs skip already-present data.
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
def _download_datasets():
    """Download all Week 1 sample datasets.

    Four datasets are needed across the practicals:
      1. karisu/General_Dataset (test_sample)  ← P1, HerdNet notebook
      2. HerdNet pretrained weights             ← HerdNet notebook
      3. Snapshot Serengeti subset              ← P3, P4
      4. Caltech Camera Traps subset            ← P3, P4, P5, P6
    """
    import sys as _sys
    from pathlib import Path as _Path

    # Resolve week1/data/ relative to this notebook file — independent of cwd
    _data = _Path(__file__).parent.parent / "data"
    if str(_data) not in _sys.path:
        _sys.path.insert(0, str(_data))

    from download_data import (
        download_general_dataset as _dl_general,
        download_herdnet_weights as _dl_weights,
        download_serengeti as _dl_serengeti,
        download_caltech as _dl_caltech,
    )

    # ── 1. HerdNet General Dataset (test sample) ────────────────────────────
    _general_ok = (
        (_data / "general_dataset" / "test_sample").exists()
        and (_data / "general_dataset" / "test_sample.csv").exists()
    )
    if _general_ok:
        _n = len([p for p in (_data / "general_dataset" / "test_sample").iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg"}])
        print(f"[1/4] General Dataset  ✓  {_n} tiles already present")
    else:
        print("[1/4] General Dataset  — downloading test sample (~200 MB) ...")
        _dl_general(full=False)
        print("      done")

    # ── 2. HerdNet pretrained weights ───────────────────────────────────────
    _weights_ok = (
        _data.parent.parent / "models" / "general_2022"
        / "20220413_HerdNet_General_dataset_2022.pth"
    ).exists()
    if _weights_ok:
        print("[2/4] HerdNet weights  ✓  already present")
    else:
        print("[2/4] HerdNet weights  — downloading (~300 MB) ...")
        _dl_weights()
        print("      done")

    # ── 3. Snapshot Serengeti subset ────────────────────────────────────────
    _serengeti_dir = _data / "camera_trap" / "serengeti_subset"
    _serengeti_ok = _serengeti_dir.exists() and any(
        p for p in _serengeti_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}
    )
    if _serengeti_ok:
        _n = len([p for p in _serengeti_dir.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg"}])
        print(f"[3/4] Serengeti        ✓  {_n} images already present")
    else:
        print("[3/4] Serengeti        — downloading 50 images ...")
        _dl_serengeti(n_images=50)
        print("      done")

    # ── 4. Caltech Camera Traps subset ──────────────────────────────────────
    _caltech_dir = _data / "camera_trap" / "caltech_subset"
    _caltech_ok = (
        _caltech_dir.exists()
        and any(p for p in _caltech_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"})
        and (_data / "camera_trap_labels.csv").exists()
    )
    if _caltech_ok:
        _n = len([p for p in _caltech_dir.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg"}])
        print(f"[4/4] Caltech          ✓  {_n} images already present")
    else:
        print("[4/4] Caltech          — downloading 50 images + labels ...")
        _dl_caltech(n_images=50)
        print("      done")

    print("\nAll datasets ready.")

    # Expose the resolved data directory so downstream cells can use absolute paths
    DATA_BASE = _data
    return (DATA_BASE,)


@app.cell
def _load(DATA_BASE, pd):
    DATA_DIR = DATA_BASE / "general_dataset" / "test_sample"
    CSV_PATH = DATA_BASE / "general_dataset" / "test_sample.csv"

    if not CSV_PATH.exists():
        print("CSV not found — check the download cell output above.")
        df = None
        tile_files = []
    else:
        df = pd.read_csv(CSV_PATH)
        tile_files = sorted(
            p for p in DATA_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}
        )
    return DATA_DIR, df, tile_files


@app.cell
def _step8(mo):
    mo.md(r"""
    ## Part 1 — Camera Trap Imagery

    Practicals 3–6 use ground-level camera trap images from two public datasets.
    We explore them here so you can see how their annotation formats differ from
    the aerial point annotations used in Part 2.

    ### Step 1 — Camera trap datasets

    | Dataset | Format | Where |
    |---------|--------|-------|
    | Snapshot Serengeti | COCO JSON — `images`, `annotations`, `categories` lists | `camera_trap/serengeti_meta.json` |
    | Caltech Camera Traps | Plain CSV — one row per image | `camera_trap_labels.csv` |

    Key difference from aerial point annotations:
    - **Image-level labels** — the whole photo has one species label, not pixel coordinates
    - **COCO JSON** separates images and annotations into separate lists linked by `image_id`
    - A Caltech CSV row has only `crop` (filename) and `true_label` — no coordinates at all
    """)
    return


@app.cell
def _load_camera_trap(DATA_BASE, pd):
    import json as _json

    _meta_path = DATA_BASE / "camera_trap" / "serengeti_meta.json"
    if _meta_path.exists():
        with open(_meta_path) as _f:
            serengeti_meta = _json.load(_f)
        _cat_map = {c["id"]: c["name"] for c in serengeti_meta["categories"]}
        _n_species = len({a["category_id"] for a in serengeti_meta["annotations"]})
        print(f"Serengeti : {len(serengeti_meta['images'])} images, {_n_species} species labels")
        print(f"  Keys in JSON: {list(serengeti_meta.keys())}")
        print(f"  Example annotation: {serengeti_meta['annotations'][0]}")
        print(f"  Example category  : {serengeti_meta['categories'][0]}")
    else:
        serengeti_meta = None
        print("Serengeti meta not found — run the download cell above")

    _caltech_path = DATA_BASE / "camera_trap_labels.csv"
    if _caltech_path.exists():
        caltech_df = pd.read_csv(_caltech_path)
        print(f"\nCaltech   : {len(caltech_df)} labelled images, "
              f"{caltech_df['true_label'].nunique()} species")
        print(f"  Columns: {list(caltech_df.columns)}")
        print(f"  First 5 rows:\n{caltech_df.head().to_string(index=False)}")
    else:
        caltech_df = None
        print("Caltech labels not found")
    return caltech_df, serengeti_meta


@app.cell
def _step9(mo):
    mo.md(r"""
    ### Step 2 — Species distribution in camera trap data
    """)
    return


@app.cell
def _camera_trap_species(caltech_df, plt, serengeti_meta):
    if serengeti_meta is None and caltech_df is None:
        print("No camera trap data loaded.")
    else:
        _fig, _axes = plt.subplots(1, 2, figsize=(14, 4))

        # Serengeti
        if serengeti_meta is not None:
            _cat_map = {c["id"]: c["name"] for c in serengeti_meta["categories"]}
            import pandas as _pd
            _labels = [_cat_map.get(a["category_id"], "unknown")
                       for a in serengeti_meta["annotations"]]
            _counts = _pd.Series(_labels).value_counts()
            _axes[0].barh(_counts.index, _counts.values, color="steelblue")
            _axes[0].set_xlabel("Number of annotations")
            _axes[0].set_title(f"Snapshot Serengeti — {len(_counts)} classes")

        # Caltech
        if caltech_df is not None:
            _counts2 = caltech_df["true_label"].value_counts()
            _axes[1].barh(_counts2.index, _counts2.values, color="darkorange")
            _axes[1].set_xlabel("Number of images")
            _axes[1].set_title(f"Caltech Camera Traps — {len(_counts2)} classes")

        plt.tight_layout()
    plt.gca()
    return


@app.cell
def _step10(mo):
    mo.md(r"""
    ### Step 3 — Example camera trap images

    Three images from each dataset, labelled by species.
    Camera trap images look very different from aerial tiles:
    fixed height, often at night, one animal fills the frame.
    """)
    return


@app.cell
def _camera_trap_gallery(
    DATA_BASE,
    Image,
    caltech_df,
    np,
    plt,
    serengeti_meta,
):
    _n_each = 3
    _fig, _axes = plt.subplots(2, _n_each, figsize=(_n_each * 4, 8))

    # ── Serengeti row ─────────────────────────────────────────────────────────
    if serengeti_meta is not None:
        _cat_map = {c["id"]: c["name"] for c in serengeti_meta["categories"]}
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
            _axes[0][_shown].imshow(np.array(Image.open(_path)))
            _axes[0][_shown].set_title(f"Serengeti\n{_label}", fontsize=9)
            _axes[0][_shown].axis("off")
            _shown += 1
        for _j in range(_shown, _n_each):
            _axes[0][_j].axis("off")

    # ── Caltech row ───────────────────────────────────────────────────────────
    if caltech_df is not None:
        _caltech_dir = DATA_BASE / "camera_trap" / "caltech_subset"
        _shown = 0
        for _, _row in caltech_df.iterrows():
            if _shown >= _n_each:
                break
            _path = _caltech_dir / _row["crop"]
            if not _path.exists():
                continue
            _axes[1][_shown].imshow(np.array(Image.open(_path)))
            _axes[1][_shown].set_title(f"Caltech\n{_row['true_label']}", fontsize=9)
            _axes[1][_shown].axis("off")
            _shown += 1
        for _j in range(_shown, _n_each):
            _axes[1][_j].axis("off")

    plt.suptitle("Camera trap examples — image-level species labels", fontsize=12)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Step 4 — Bounding box annotations

    Caltech Camera Traps includes **ground-truth bounding boxes** stored as
    `[x, y, w, h]` in pixel coordinates (top-left corner, width, height) — the
    **COCO format**. Not every image in our 50-image sample has a box; the bbox
    file is a separate download, so coverage is partial.

    Bounding boxes are the annotation type MegaDetector outputs (Practical 3).
    Seeing ground-truth boxes now helps you judge detector quality later.

    You will learn about different bbox formats (COCO, YOLO, Pascal VOC) and
    how to convert between them in **Practical 3, Step 9** when you prepare
    YOLO training labels.
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
            _ax.set_title(
                f"{_row['true_label']}\n"
                f"({int(_row['bbox_x'])}, {int(_row['bbox_y'])}) "
                f"{int(_row['bbox_w'])}x{int(_row['bbox_h'])} px",
                fontsize=8,
            )
            _ax.axis("off")
            _shown += 1

        for _ax in _axes_flat[_shown:]:
            _ax.axis("off")

        plt.suptitle(
            f"Ground-truth bounding boxes — COCO [x, y, w, h] format  "
            f"({len(_with_box)}/{len(caltech_df)} images have boxes)",
            fontsize=11,
        )
        plt.tight_layout()
    plt.gca()
    return


@app.cell
def _step1(mo):
    mo.md(r"""
    ## Part 2 — Aerial Point Annotations (HerdNet)

    The HerdNet General Dataset contains pre-tiled aerial images of African mammals
    with **point annotations** — one `(x, y)` coordinate per animal, no bounding boxes.
    This is the dataset format used throughout Week 1 for the iguana case study.

    ### Step 1 — What is in the CSV?

    The annotation file is a plain CSV with four columns.
    Every row is one animal. `x` and `y` are pixel coordinates within the tile.
    """)
    return


@app.cell
def _csv_inspect(df, tile_files):
    if df is not None:
        print(f"Tiles in folder  : {len(tile_files)}")
        print(f"Total annotations: {len(df)}")
        print(f"Annotated tiles  : {df['images'].nunique()}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nFirst 10 rows:")
        print(df.head(10).to_string(index=False))
    return


@app.cell
def _step2(mo):
    mo.md(r"""
    ### Step 2 — Class distribution

    How many annotations are there per species?
    """)
    return


@app.cell
def _classes(df, plt):
    if df is not None:
        _counts = df["labels"].value_counts()
        print("Class counts:")
        print(_counts.to_string())

        _fig, _ax = plt.subplots(figsize=(7, 3))
        _ax.barh(_counts.index, _counts.values, color="steelblue")
        _ax.set_xlabel("Number of annotations")
        _ax.set_title("Annotations per species class")
        plt.tight_layout()
    plt.gca()
    return


@app.cell
def _step3(mo):
    mo.md(r"""
    ### Step 3 — Annotations per tile

    Not all tiles have animals. How are annotations distributed across tiles?
    """)
    return


@app.cell
def _tile_dist(df, plt):
    if df is not None:
        _per_tile = df.groupby("images").size().sort_values(ascending=False)
        print(f"Tiles with annotations : {len(_per_tile)}")
        print(f"Max per tile           : {_per_tile.max()}")
        print(f"Median per tile        : {_per_tile.median():.0f}")
        print(f"Min per tile (in CSV)  : {_per_tile.min()}")

        _fig, _ax = plt.subplots(figsize=(7, 3))
        _ax.hist(_per_tile.values, bins=20, color="steelblue", edgecolor="white")
        _ax.set_xlabel("Annotations per tile")
        _ax.set_ylabel("Number of tiles")
        _ax.set_title("Distribution of annotation counts across tiles")
        plt.tight_layout()
    plt.gca()
    return


@app.cell
def _step4(mo):
    mo.md(r"""
    ### Step 4 — What does a tile look like?

    Each tile is a plain JPEG. Open one and look at its size, colour mode, and pixel values.
    """)
    return


@app.cell
def _tile_inspect(Image, np, tile_files):
    if tile_files:
        _path = tile_files[0]
        _img = Image.open(_path)
        _arr = np.array(_img)
        print(f"File  : {_path.name}")
        print(f"Size  : {_img.width} x {_img.height} px")
        print(f"Mode  : {_img.mode}")
        print(f"dtype : {_arr.dtype}   shape: {_arr.shape}")
        print()
        for _i, _ch in enumerate(["R", "G", "B"]):
            print(f"  {_ch}: min={_arr[:,:,_i].min()}  max={_arr[:,:,_i].max()}  mean={_arr[:,:,_i].mean():.1f}")
    return


@app.cell
def _step5(mo):
    mo.md(r"""
    ### Step 5 — Tiles with annotations

    The three tiles with the most animals. Red + marks = one animal each.
    """)
    return


@app.cell
def _gallery(DATA_DIR, Image, df, np, plt, tile_files):
    if tile_files and df is not None:
        _counts = df.groupby("images").size().sort_values(ascending=False)
        _top3 = _counts.head(3).index.tolist()

        _fig, _axes = plt.subplots(1, 3, figsize=(15, 5))
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
    plt.gca()
    return


@app.cell
def _step6(mo):
    mo.md(r"""
    ### Step 6 — The annotation format in detail

    The CSV has four columns:

    | Column | Type | Meaning |
    |--------|------|---------|
    | `images` | string | JPEG filename of the tile (e.g. `tile_0042.jpg`) |
    | `x` | int | Horizontal pixel coordinate **within the tile** (column, left→right) |
    | `y` | int | Vertical pixel coordinate **within the tile** (row, top→bottom) |
    | `labels` | string | Species class name |

    Key points:
    - **One row = one animal.** A tile with 12 animals has 12 rows in the CSV.
    - **Empty tiles are absent** — a tile with no animals simply has no rows at all.
    - Coordinates are **pixel-relative** to the tile's top-left corner `(0, 0)`.
      They are not GPS coordinates. To geo-locate an annotation you also need the
      tile's offset within the full orthomosaic and the image's ground sampling distance.
    - The `labels` column is a class name assigned by a human annotator
      (e.g. `"buffalo"`, `"zebra"`, `"elephant"`).
    """)
    return


@app.cell
def _format_detail(DATA_DIR, Image, df, np, plt, tile_files):
    """Animal crops from the most-annotated tile, shown in a grid."""
    CROP_RADIUS = 32   # pixels around each point annotation
    N_CROPS = 24       # max crops to display

    if tile_files and df is not None:
        _counts = df.groupby("images").size().sort_values(ascending=False)
        _name = _counts.index[0]
        _tile_df = df[df["images"] == _name].copy()

        _img = np.array(Image.open(DATA_DIR / _name))
        _H, _W = _img.shape[:2]

        _classes = sorted(_tile_df["labels"].unique())
        _colors = plt.cm.tab10.colors
        _colour_map = {cls: _colors[i % len(_colors)] for i, cls in enumerate(_classes)}

        # One crop per annotation point (up to N_CROPS)
        _sample = _tile_df.head(N_CROPS)
        _n = len(_sample)
        _ncols = 8
        _nrows = (_n + _ncols - 1) // _ncols

        _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(_ncols * 1.6, _nrows * 1.6))
        _axes = np.array(_axes).flatten()

        for _i, (_, _row) in enumerate(_sample.iterrows()):
            _cx, _cy = int(_row["x"]), int(_row["y"])
            _x0 = max(0, _cx - CROP_RADIUS)
            _x1 = min(_W, _cx + CROP_RADIUS)
            _y0 = max(0, _cy - CROP_RADIUS)
            _y1 = min(_H, _cy + CROP_RADIUS)
            _crop = _img[_y0:_y1, _x0:_x1]

            _axes[_i].imshow(_crop)
            _axes[_i].set_title(
                _row["labels"], fontsize=6,
                color=_colour_map[_row["labels"]], pad=2,
            )
            _axes[_i].axis("off")

        for _ax in _axes[_n:]:
            _ax.axis("off")

        plt.suptitle(
            f"Animal crops from {_name}  (r={CROP_RADIUS} px around each point annotation)",
            fontsize=10,
        )
        plt.tight_layout()
    plt.gca()
    return


@app.cell
def _step7(mo):
    mo.md(r"""
    ### Step 7 — One representative tile per species

    For each class, the tile with the **most animals of that species**.
    This shows you what each species looks like from above at drone resolution.
    """)
    return


@app.cell
def _species_gallery(DATA_DIR, Image, df, np, plt):
    """One tile per species class — the richest tile for each class."""
    if df is not None:
        _classes = sorted(df["labels"].unique())
        _n = len(_classes)
        _ncols = min(4, _n)
        _nrows = (_n + _ncols - 1) // _ncols

        _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(_ncols * 4, _nrows * 4))
        _axes_flat = np.array(_axes).flatten()

        for _i, _cls in enumerate(_classes):
            _sub = df[df["labels"] == _cls]
            _best_tile = _sub.groupby("images").size().idxmax()
            _tile_anns = _sub[_sub["images"] == _best_tile]

            _img = np.array(Image.open(DATA_DIR / _best_tile))
            _ax = _axes_flat[_i]
            _ax.imshow(_img)
            _ax.scatter(
                _tile_anns["x"], _tile_anns["y"],
                c="red", s=25, marker="+", linewidths=1.5,
            )
            _ax.set_title(f"{_cls}\n({len(_tile_anns)} animals)", fontsize=9)
            _ax.axis("off")

        for _ax in _axes_flat[_n:]:
            _ax.axis("off")

        plt.suptitle("Richest tile per species class", fontsize=13)
        plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Step 8 — Dense vs empty tiles

    Most tiles in a drone survey have **no animals at all** — they just show habitat.
    The CSV only contains rows for annotated tiles, so empty tiles are invisible in
    the data. Below: the most annotated tile alongside one of the empty tiles.
    """)
    return


@app.cell
def _(DATA_DIR, Image, df, np, plt, tile_files):
    if tile_files and df is not None:
        # Most annotated tile
        _annotated_names = set(df["images"].unique())
        _dense_name = df.groupby("images").size().idxmax()
        _dense_img = np.array(Image.open(DATA_DIR / _dense_name))
        _dense_ann = df[df["images"] == _dense_name]

        # First tile not in the CSV at all → truly empty
        _empty_path = next(
            (p for p in tile_files if p.name not in _annotated_names), None
        )

        _fig, _axes = plt.subplots(1, 2, figsize=(12, 5))

        _axes[0].imshow(_dense_img)
        _axes[0].scatter(
            _dense_ann["x"], _dense_ann["y"],
            c="red", s=30, marker="+", linewidths=1.5,
        )
        _axes[0].set_title(
            f"Dense tile: {_dense_name}\n({len(_dense_ann)} annotations)", fontsize=10
        )
        _axes[0].axis("off")

        if _empty_path:
            _axes[1].imshow(np.array(Image.open(_empty_path)))
            _axes[1].set_title(
                f"Empty tile: {_empty_path.name}\n(0 annotations — absent from CSV)",
                fontsize=10,
            )
        else:
            _axes[1].set_title("No empty tiles found in folder", fontsize=10)
        _axes[1].axis("off")

        plt.suptitle("Dense vs empty tile comparison", fontsize=12)
        plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Step 9 — Annotation positions across all tiles

    Where do animals appear within tiles? A scatter plot of all `(x, y)` annotation
    coordinates, coloured by species. Each dot is one animal across all tiles.
    Clusters reveal where annotators tended to find animals — usually away from the
    tile edges and not at the very top (sky) or bottom (ground shadow).
    """)
    return


@app.cell
def _(df, plt):
    if df is not None:
        _classes = sorted(df["labels"].unique())
        _colors = plt.cm.tab10.colors

        _fig, _ax = plt.subplots(figsize=(7, 7))
        for _i, _cls in enumerate(_classes):
            _sub = df[df["labels"] == _cls]
            _ax.scatter(
                _sub["x"], _sub["y"],
                color=_colors[_i % len(_colors)],
                s=8, alpha=0.5, label=_cls,
            )
        _ax.set_xlabel("x (pixels within tile)")
        _ax.set_ylabel("y (pixels within tile)")
        _ax.invert_yaxis()  # image coords: y=0 at top
        _ax.legend(fontsize=8, markerscale=2)
        _ax.set_title(
            f"All annotation positions — {len(df)} animals across "
            f"{df['images'].nunique()} tiles",
            fontsize=11,
        )
        plt.tight_layout()
    plt.gca()
    return


@app.cell
def _exercise(mo):
    mo.md(r"""
    ## Exercise

    1. Which species is most common in this dataset? Which is rarest?
    2. How many tiles have **more than 10** annotations? What fraction of all annotated tiles is that?
    3. Look at the three tiles shown above. Can you spot all the annotated animals?
       Are any annotations clearly wrong?
    4. The annotations are stored as pixel coordinates. What extra information would you need
       to convert them to GPS positions?
    """)
    return


@app.cell
def _reflection(mo):
    mo.md(r"""
    ## Reflection

    - Tiles without any annotations are **not listed in the CSV** — they simply have no rows.
      What does this mean for counting "empty" tiles?
    - If your survey covers a large area, most tiles will be empty. How should the training
      pipeline handle this imbalance?
    - The coordinates `x`, `y` are relative to the tile, not the full orthomosaic.
      What would you need to place each annotation back into the original image?
    """)
    return


if __name__ == "__main__":
    app.run()
