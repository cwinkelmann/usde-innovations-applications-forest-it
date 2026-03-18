import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


### Reference: https://colab.research.google.com/github/agentmorris/MegaDetector/blob/main/notebooks/megadetector_colab.ipynb
### Postprocessing (separate into folders): https://github.com/agentmorris/MegaDetector/tree/main/megadetector/postprocessing


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _imports():
    from pathlib import Path

    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image

    DATA_DIR = Path(__file__).parent.parent / "data"

    return DATA_DIR, Image, mpatches, np, pd, plt


@app.cell
def _helpers(mpatches):
    def draw_boxes(ax, boxes_xyxy, labels, scores, color="#E74C3C"):
        """Draw bounding boxes on a matplotlib axis.

        Args:
            ax: matplotlib Axes
            boxes_xyxy: list of [x1, y1, x2, y2] in pixel coordinates
            labels: list of class name strings
            scores: list of confidence floats
            color: single colour string or list of colours per box
        """
        for i, ((x1, y1, x2, y2), lbl, sc) in enumerate(
            zip(boxes_xyxy, labels, scores)
        ):
            c = color[i] if isinstance(color, list) else color
            ax.add_patch(mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5, edgecolor=c, facecolor="none",
            ))
            ax.text(
                x1, y1 - 3, f"{lbl} {sc:.2f}",
                fontsize=7, color=c, fontweight="bold",
            )

    return (draw_boxes,)


@app.cell(hide_code=True)
def _context(mo):
    mo.md(r"""
    # Practical 3 — Animal Detection with MegaDetector

    **Context:** A camera trap deployment produces thousands of images per week.
    More than 80 % are empty — triggered by wind or vegetation movement.
    MegaDetector is a pre-trained detector that filters those empty frames automatically,
    reducing manual review by ~80 %.

    MegaDetector detects three classes: **animal · person · vehicle**.
    It does *not* identify species — species classification is a downstream task (P5).

    **MegaDetector versions:**

    | Version | Architecture | Weights | How to load |
    |---------|-------------|---------|-------------|
    | MDv5a/b | YOLOv5x6 | ~600 MB | `megadetector` package |
    | MD v1000 larch | YOLOv11-L | 49 MB | `ultralytics.YOLO()` |
    | MD v1000 sorrel | YOLOv11-S | 18 MB | `ultralytics.YOLO()` |
    | MDV6-rtdetr-c | RT-DETR-L | 66 MB | `ultralytics.RTDETR()` |

    In this practical we use the **ultralytics** API directly — no special
    `megadetector` package needed. All MD v1000 models are standard `.pt` files
    from [github.com/agentmorris/MegaDetector/releases](https://github.com/agentmorris/MegaDetector/releases).

    **Install:**
    ```bash
    pip install ultralytics
    ```
    """)
    return


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — MEGADETECTOR INFERENCE
# ═══════════════════════════════════════════════════════════════════════════


@app.cell(hide_code=True)
def _part1_header(mo):
    mo.md(r"""
    ## Part 1 — Inference with MegaDetector v1000

    We load `md_v1000.0.0-larch.pt` — a YOLOv11-L model (25M params) that
    runs through the standard ultralytics `YOLO()` class.

    The model auto-downloads from GitHub on first use (~49 MB).
    """)
    return


@app.cell
def _load_model():
    import os
    import torch
    import wget
    from ultralytics import YOLO

    # Auto-download larch weights if not cached
    cache_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    weights_path = os.path.join(cache_dir, "md_v1000.0.0-larch.pt")
    if not os.path.exists(weights_path):
        os.makedirs(cache_dir, exist_ok=True)
        print("Downloading MD v1000 larch weights...")
        wget.download(
            "https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-larch.pt",
            out=weights_path,
        )
        print()

    md_model = YOLO(weights_path)
    print(f"MegaDetector v1000 (larch) loaded")
    print(f"  Architecture : YOLOv11-L")
    print(f"  Classes      : {md_model.names}")
    print(f"  Weights      : {weights_path}")

    MD_LABELS = md_model.names  # {0: 'animal', 1: 'person', 2: 'vehicle'}
    MD_COLORS = {0: "#E74C3C", 1: "#3498DB", 2: "#F39C12"}

    return MD_COLORS, MD_LABELS, md_model


@app.cell(hide_code=True)
def _step2_md(mo):
    mo.md(r"""
    ### Step 2 — Single-image detection

    `model.predict()` returns a list of `Results` objects. Each result has:
    - `result.boxes.xyxy` — bounding boxes in `[x1, y1, x2, y2]` pixel coordinates
    - `result.boxes.conf` — confidence scores
    - `result.boxes.cls` — class indices (0=animal, 1=person, 2=vehicle)

    Unlike the old MDv5 API (which returns normalised `[x, y, w, h]`), ultralytics
    returns pixel coordinates directly — no manual conversion needed.
    """)
    return


@app.cell
def _single_image(DATA_DIR, Image, MD_COLORS, MD_LABELS, draw_boxes, md_model, np, plt):
    _img_dir = DATA_DIR / "camera_trap" / "serengeti_subset"
    _img_files = sorted(
        p for p in _img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}
    ) if _img_dir.exists() else []

    if not _img_files:
        print("No Serengeti images found.")
        print("Run: python scripts/data/download_data.py --sample")
    else:
        _img_path = _img_files[0]
        _pil = Image.open(_img_path).convert("RGB")
        _arr = np.array(_pil)

        # Standard ultralytics inference
        _results = md_model.predict(_arr, conf=0.1, verbose=False)
        _boxes_obj = _results[0].boxes

        print(f"Image : {_img_path.name}  ({_pil.width}x{_pil.height} px)")
        print(f"Hits  : {len(_boxes_obj)}")
        for _i in range(len(_boxes_obj)):
            _cls_id = int(_boxes_obj.cls[_i])
            _conf = float(_boxes_obj.conf[_i])
            _x1, _y1, _x2, _y2 = _boxes_obj.xyxy[_i].tolist()
            print(f"  {MD_LABELS[_cls_id]:8s}  conf={_conf:.3f}  "
                  f"box=[{_x1:.0f}, {_y1:.0f}, {_x2:.0f}, {_y2:.0f}]")

        # Visualize
        _fig, _ax = plt.subplots(figsize=(10, 7))
        _ax.imshow(_arr)
        _boxes = [_boxes_obj.xyxy[_j].tolist() for _j in range(len(_boxes_obj))]
        _labels = [MD_LABELS[int(_boxes_obj.cls[_j])] for _j in range(len(_boxes_obj))]
        _scores = [float(_boxes_obj.conf[_j]) for _j in range(len(_boxes_obj))]
        _colors = [MD_COLORS[int(_boxes_obj.cls[_j])] for _j in range(len(_boxes_obj))]
        draw_boxes(_ax, _boxes, _labels, _scores, color=_colors)
        _ax.set_title(f"MegaDetector v1000 larch — {_img_path.name}", fontsize=10)
        _ax.axis("off")
        plt.tight_layout()

    plt.gca()
    return


@app.cell(hide_code=True)
def _step3_md(mo):
    mo.md(r"""
    ### Step 3 — Batch detection on Serengeti images

    Run MegaDetector on all downloaded Serengeti images and collect results
    into a DataFrame. This is the starting point for the two-stage pipeline:

    ```
    images  →  MegaDetector  →  animal crops  →  species classifier (P5)
    ```
    """)
    return


@app.cell
def _batch(DATA_DIR, Image, MD_LABELS, md_model, np, pd):
    _CONF_THRESHOLD = 0.2

    _sources = {
        "serengeti": DATA_DIR / "camera_trap" / "serengeti_subset",
        "caltech":   DATA_DIR / "camera_trap" / "caltech_subset",
    }

    records = []
    _total_images = 0
    for _source, _img_dir in _sources.items():
        if not _img_dir.exists():
            continue
        _files = sorted(
            p for p in _img_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg"}
        )
        for _p in _files:
            _total_images += 1
            _results = md_model.predict(str(_p), conf=_CONF_THRESHOLD, verbose=False)
            _boxes = _results[0].boxes
            _W, _H = Image.open(_p).size

            if len(_boxes) == 0:
                # Record empty image
                records.append({
                    "source":     _source,
                    "filename":   _p.name,
                    "filepath":   str(_p),
                    "category":   "empty",
                    "confidence": 0.0,
                    "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 0, "bbox_y2": 0,
                })
            else:
                for _bi in range(len(_boxes)):
                    _bx1, _by1, _bx2, _by2 = _boxes.xyxy[_bi].tolist()
                    records.append({
                        "source":     _source,
                        "filename":   _p.name,
                        "filepath":   str(_p),
                        "category":   MD_LABELS[int(_boxes.cls[_bi])],
                        "confidence": round(float(_boxes.conf[_bi]), 4),
                        "bbox_x1": round(_bx1, 1),
                        "bbox_y1": round(_by1, 1),
                        "bbox_x2": round(_bx2, 1),
                        "bbox_y2": round(_by2, 1),
                    })

    detections_df = pd.DataFrame(records)
    _out = DATA_DIR / "camera_trap_detections.csv"
    detections_df.to_csv(_out, index=False)

    _empty = len(detections_df[detections_df["category"] == "empty"])
    _with_det = _total_images - _empty

    print(f"Images processed : {_total_images}")
    print(f"  With detections: {_with_det}")
    print(f"  Empty (no hits) : {_empty}")
    print(f"Total detections : {len(detections_df[detections_df['category'] != 'empty'])}")
    print()
    print("Category breakdown:")
    print(detections_df[detections_df["category"] != "empty"]
          .groupby(["source", "category"]).size().to_string())
    print(f"\nSaved → {_out}")

    return (detections_df,)


@app.cell(hide_code=True)
def _step4_md(mo):
    mo.md(r"""
    ### Step 4 — Confidence threshold and empty-frame filtering

    The confidence threshold is the most important tuning parameter.
    Too low → many false positives (wind, leaves). Too high → missed animals.

    A common field deployment strategy: set threshold at 0.2, send all
    detections above it to a reviewer, skip everything below.

    **The histogram below shows the confidence distribution of all detections.**
    """)
    return


@app.cell
def _threshold_analysis(detections_df, np, plt):
    _det = detections_df[detections_df["category"] != "empty"]

    if len(_det) == 0:
        print("No detections to plot.")
    else:
        _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))

        _axes[0].hist(
            _det["confidence"], bins=30,
            color="steelblue", edgecolor="white",
        )
        _axes[0].set_xlabel("Confidence score")
        _axes[0].set_ylabel("Number of detections")
        _axes[0].set_title("Confidence distribution — all detections")
        for _t in [0.2, 0.5, 0.8]:
            _axes[0].axvline(_t, color="red", linestyle="--", linewidth=1, alpha=0.6,
                             label=f"t={_t}")
        _axes[0].legend(fontsize=8)

        _thresholds = np.arange(0.05, 0.99, 0.05)
        _counts = [len(_det[_det["confidence"] >= t]) for t in _thresholds]
        _axes[1].plot(_thresholds, _counts, "o-", color="darkorange")
        _axes[1].set_xlabel("Confidence threshold")
        _axes[1].set_ylabel("Detections retained")
        _axes[1].set_title("Detections retained vs. threshold")
        _axes[1].grid(alpha=0.3)

        plt.suptitle("Choosing a confidence threshold", fontsize=12)
        plt.tight_layout()

    plt.gca()
    return


@app.cell(hide_code=True)
def _step5_md(mo):
    mo.md(r"""
    ### Step 5 — Separate images into folders

    The classic MegaDetector workflow sorts images into
    `animal/`, `person/`, `vehicle/`, and `empty/` folders based on the
    highest-confidence detection per image. This lets field teams quickly
    review only the animal folder.

    See: [agentmorris/MegaDetector postprocessing](https://github.com/agentmorris/MegaDetector/tree/main/megadetector/postprocessing)
    """)
    return


@app.cell
def _sort_into_folders(DATA_DIR, detections_df):
    import shutil

    _sort_dir = DATA_DIR / "camera_trap_sorted"

    # Get highest-confidence category per image
    _det = detections_df.copy()
    _best = _det.sort_values("confidence", ascending=False).drop_duplicates("filename", keep="first")

    _counts = {}
    for _, _row in _best.iterrows():
        _cat = _row["category"]
        _dst_dir = _sort_dir / _cat
        _dst_dir.mkdir(parents=True, exist_ok=True)
        _src = _row["filepath"]
        _dst = _dst_dir / _row["filename"]
        if not _dst.exists() and Path(_src).exists():
            shutil.copy2(_src, _dst)
        _counts[_cat] = _counts.get(_cat, 0) + 1

    print(f"Sorted into: {_sort_dir}")
    for _cat, _n in sorted(_counts.items()):
        print(f"  {_cat:10s}: {_n} images")

    return


@app.cell(hide_code=True)
def _step6_md(mo):
    mo.md(r"""
    ### Step 6 — Extract animal crops

    Save each animal detection as a JPEG crop.
    These crops are the input to the species classifier in P5.
    """)
    return


@app.cell
def _crops(DATA_DIR, Image, detections_df):
    _crops_dir = DATA_DIR / "camera_trap_crops"
    _crops_dir.mkdir(exist_ok=True)

    _animals = detections_df[
        (detections_df["category"] == "animal") &
        (detections_df["confidence"] >= 0.2)
    ].copy()

    _saved = 0
    for _, _row in _animals.iterrows():
        _img = Image.open(_row["filepath"]).convert("RGB")
        _crop = _img.crop((
            int(_row["bbox_x1"]), int(_row["bbox_y1"]),
            int(_row["bbox_x2"]), int(_row["bbox_y2"]),
        ))
        _out = _crops_dir / f"{_row['source']}_{_row['filename'].rsplit('.', 1)[0]}_crop{_saved:04d}.jpg"
        _crop.save(_out, quality=90)
        _saved += 1

    print(f"Saved {_saved} animal crops → {_crops_dir}")
    return


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — SAHI: TILED INFERENCE FOR LARGE IMAGES
# ═══════════════════════════════════════════════════════════════════════════


@app.cell(hide_code=True)
def _part2_header(mo):
    mo.md(r"""
    ## Part 2 — SAHI: Tiled Inference for Large Images

    MegaDetector was trained on camera trap images (~1920x1080 px).
    Drone orthomosaics can be 10,000x10,000 px or larger, with animals
    appearing as tiny blobs of 20-50 pixels.

    **SAHI** (Slicing Aided Hyper Inference) solves this by:

    1. Slicing the image into overlapping tiles (e.g. 640x640)
    2. Running detection on each tile independently
    3. Merging results with NMS to remove duplicates at tile borders

    **But do you always need SAHI?** Our benchmarks show that running
    the model at higher input resolution (e.g. 2560-4096px) without tiling
    can actually outperform SAHI — if the GPU has enough memory.

    | Approach | F1 on Eikelboom test | VRAM |
    |----------|---------------------|------|
    | SAHI 640px tiles | 0.546 | ~2 GB |
    | Direct at 2560px | 0.656 | ~0.6 GB |
    | Direct at 4096px | **0.721** | ~2.4 GB |

    ```
    ┌──────────────────────────┐
    │  ┌───────┬───┬───────┐   │
    │  │ tile1 │ ↔ │ tile2 │   │   ↔ = overlap zone
    │  ├───────┼───┼───────┤   │
    │  │   ↕   │   │   ↕   │   │   ↕ = overlap zone
    │  ├───────┼───┼───────┤   │
    │  │ tile3 │   │ tile4 │   │
    │  └───────┴───┴───────┘   │
    └──────────────────────────┘
    ```
    """)
    return


@app.cell(hide_code=True)
def _step7_md(mo):
    mo.md(r"""
    ### Step 7 — Full-image vs. sliced inference

    We compare standard full-image detection with SAHI's sliced approach
    on a drone image — using the same MegaDetector model.

    On large images with small targets, sliced inference typically finds
    significantly more objects because each tile is resized to the model's
    native resolution (640 px), making small objects appear larger.
    """)
    return


@app.cell
def _sahi_compare(DATA_DIR, Image, draw_boxes, md_model, np, plt):
    from sahi import AutoDetectionModel
    from sahi.predict import get_prediction, get_sliced_prediction

    # Wrap our ultralytics model for SAHI
    import os as _os
    import torch as _torch
    _weights_path = _os.path.join(_torch.hub.get_dir(), "checkpoints", "md_v1000.0.0-larch.pt")
    sahi_det_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=_weights_path,
        confidence_threshold=0.2,
        device="cuda:0" if _torch.cuda.is_available() else "cpu",
    )

    _drone_dir = DATA_DIR / "general_dataset" / "test_sample"
    _drone_imgs = sorted(_drone_dir.glob("*.jpeg"))[:1] if _drone_dir.exists() else []

    if not _drone_imgs:
        # Fallback: use an Eikelboom test image
        _eik_dir = DATA_DIR / "eikelboom" / "test"
        _drone_imgs = sorted(_eik_dir.glob("*.JPG"))[:1] if _eik_dir.exists() else []

    if not _drone_imgs:
        print("No drone images found.")
        print("Run: python scripts/data/download_data.py --sample")
    else:
        _img_path = str(_drone_imgs[0])
        _pil = Image.open(_img_path).convert("RGB")
        _arr = np.array(_pil)

        _full = get_prediction(
            image=_img_path,
            detection_model=sahi_det_model,
        )

        _sliced = get_sliced_prediction(
            image=_img_path,
            detection_model=sahi_det_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            postprocess_type="NMS",
            postprocess_match_threshold=0.5,
        )

        print(f"Image: {_drone_imgs[0].name} ({_pil.width}x{_pil.height} px)")
        print(f"Full-image detections : {len(_full.object_prediction_list)}")
        print(f"Sliced detections     : {len(_sliced.object_prediction_list)}")

        _fig, _axes = plt.subplots(1, 2, figsize=(14, 6))

        for _ax, _res, _title in [
            (_axes[0], _full, "Full-image inference"),
            (_axes[1], _sliced, "SAHI sliced (640x640, 20% overlap)"),
        ]:
            _ax.imshow(_arr)
            _boxes = [(p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy)
                      for p in _res.object_prediction_list]
            _labels = [p.category.name for p in _res.object_prediction_list]
            _scores = [p.score.value for p in _res.object_prediction_list]
            draw_boxes(_ax, _boxes, _labels, _scores)
            _ax.set_title(f"{_title} ({len(_boxes)} hits)", fontsize=9)
            _ax.axis("off")

        plt.suptitle(
            "Full-image vs. SAHI tiled inference on drone imagery", fontsize=11
        )
        plt.tight_layout()

    plt.gca()
    return get_sliced_prediction, sahi_det_model


@app.cell(hide_code=True)
def _step8_md(mo):
    mo.md(r"""
    ### Step 8 — Interactive SAHI parameter tuning

    Use the sliders below to explore how slice size, overlap, and confidence
    threshold affect detection results.

    | Parameter | ↑ Higher | ↓ Lower |
    |-----------|----------|---------|
    | **Slice size** | Faster, misses small objects | Slower, catches tiny targets |
    | **Overlap** | Better border coverage, slower | Faster, may miss border objects |
    | **Confidence** | Fewer false positives | More detections, more noise |

    **Postprocess strategies:**
    - **NMS** (Non-Maximum Suppression) — removes the lower-confidence box when two overlap
    - **NMM** (Non-Maximum Merging) — merges overlapping boxes into one larger box
    """)
    return


@app.cell
def _sahi_sliders(mo):
    slice_size = mo.ui.slider(
        128, 1024, step=64, value=640, label="Slice size (px)"
    )
    overlap = mo.ui.slider(
        0.0, 0.5, step=0.05, value=0.2, label="Overlap ratio"
    )
    sahi_conf = mo.ui.slider(
        0.05, 0.9, step=0.05, value=0.2, label="Confidence threshold"
    )
    postprocess = mo.ui.dropdown(
        options={"NMS (suppress duplicates)": "NMS", "NMM (merge duplicates)": "NMM"},
        value="NMS",
        label="Postprocess",
    )
    mo.hstack([slice_size, overlap, sahi_conf, postprocess], justify="start", gap=2)
    return overlap, postprocess, sahi_conf, slice_size


@app.cell
def _sahi_interactive(
    DATA_DIR, Image, draw_boxes, get_sliced_prediction, np, overlap, plt, postprocess,
    sahi_conf, sahi_det_model, slice_size,
):

    _drone_dir = DATA_DIR / "general_dataset" / "test_sample"
    _drone_imgs = sorted(_drone_dir.glob("*.jpeg"))[:1] if _drone_dir.exists() else []
    if not _drone_imgs:
        _eik_dir = DATA_DIR / "eikelboom" / "test"
        _drone_imgs = sorted(_eik_dir.glob("*.JPG"))[:1] if _eik_dir.exists() else []

    if not _drone_imgs:
        print("No drone images found.")
    else:
        _img_path = str(_drone_imgs[0])
        _pil = Image.open(_img_path).convert("RGB")
        _arr = np.array(_pil)

        sahi_det_model.confidence_threshold = sahi_conf.value

        _result = get_sliced_prediction(
            image=_img_path,
            detection_model=sahi_det_model,
            slice_height=slice_size.value,
            slice_width=slice_size.value,
            overlap_height_ratio=overlap.value,
            overlap_width_ratio=overlap.value,
            postprocess_type=postprocess.value,
            postprocess_match_threshold=0.5,
        )

        _stride = max(1, int(slice_size.value * (1 - overlap.value)))
        _n_tiles = (
            ((_pil.width - 1) // _stride + 1) *
            ((_pil.height - 1) // _stride + 1)
        )

        print(f"Image : {_drone_imgs[0].name} ({_pil.width}x{_pil.height} px)")
        print(f"Tiles : ~{_n_tiles}  (slice={slice_size.value}, overlap={overlap.value:.0%})")
        print(f"Hits  : {len(_result.object_prediction_list)}")

        _fig, _ax = plt.subplots(figsize=(10, 8))
        _ax.imshow(_arr)
        _boxes = [(p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy)
                  for p in _result.object_prediction_list]
        _labels = [p.category.name for p in _result.object_prediction_list]
        _scores = [p.score.value for p in _result.object_prediction_list]
        draw_boxes(_ax, _boxes, _labels, _scores)

        _ax.set_title(
            f"SAHI — slice={slice_size.value}px, overlap={overlap.value:.0%}, "
            f"conf>={sahi_conf.value:.2f}, {postprocess.value} → {len(_boxes)} detections",
            fontsize=10,
        )
        _ax.axis("off")
        plt.tight_layout()

    plt.gca()
    return


# ═══════════════════════════════════════════════════════════════════════════
# EXERCISES & REFLECTION
# ═══════════════════════════════════════════════════════════════════════════


@app.cell(hide_code=True)
def _exercise(mo):
    mo.md(r"""
    ## Exercise

    1. **Threshold sweep** — Re-run Step 3 with `_CONF_THRESHOLD` at 0.05, 0.2, and 0.5.
       At each threshold, how many images are classified as "empty"?

    2. **Failure modes** — Browse your crops in `data/camera_trap_crops/`.
       Find one of each:
       - A confident correct detection (conf > 0.8)
       - A confident false positive (conf > 0.5 but clearly not an animal)
       - A missed animal (visible in the original image, no detection)

    3. **SAHI parameter exploration** — Using the sliders in Step 8:
       - Set slice size to 256 and then 1024. How does detection count change?
       - Set overlap to 0 vs 0.4. Do you see missed objects at tile borders?
       - Switch postprocess from NMS to NMM. What happens to overlapping detections?

    4. **Resolution experiment** — Try running the model directly at different
       input sizes instead of using SAHI:
       ```python
       results = md_model.predict("drone_image.jpg", imgsz=2560, half=True)
       ```
       How does detection quality compare to SAHI at 640px tiles?

    5. **Compare model sizes** — Load `md_v1000.0.0-sorrel.pt` (9M params, 18MB)
       instead of larch (25M, 49MB). How does detection quality change?
       When would you prefer the smaller model?
    """)
    return


@app.cell(hide_code=True)
def _reflection(mo):
    mo.md(r"""
    ## Reflection

    - MegaDetector was trained on millions of camera trap images worldwide.
      It detects *animal* as one class — not species. What are the implications
      for a monitoring pipeline? Where does species classification fit in?

    - SAHI adds a tiling layer on top of any detector. When is this essential
      (drone surveys, satellite imagery) vs unnecessary (camera traps at 1080p)?
      What is the computational cost?

    - Our benchmarks showed that running the model at higher resolution (4096px)
      without SAHI actually beat SAHI at 640px tiles. Why might this be the case?
      When would SAHI still be preferable?

    - MegaDetector v1000 larch (49 MB) replaces MDv5a (600 MB) with similar accuracy.
      What changed between 2022 (YOLOv5) and 2025 (YOLOv11)?

    - Compare two approaches for detecting small animals in large drone images:
      (a) SAHI + MegaDetector (box-based, tile-and-merge)
      (b) HerdNet (density-map-based, with Stitcher)
      What are the trade-offs?
    """)
    return


if __name__ == "__main__":
    app.run()
