# -*- coding: utf-8 -*-
import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


# -- Shared imports ------------------------------------------------------------

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
    import torch
    from PIL import Image

    DATA_DIR = Path(__file__).parent.parent / "data"

    return DATA_DIR, Image, mpatches, np, pd, plt, torch


# -- Markdown ------------------------------------------------------------------

@app.cell(hide_code=True)
def _context(mo):
    mo.md(r"""
    # Practical 3 -- Animal Detection with MegaDetector

    **Context:** A camera trap deployment produces thousands of images per week.
    More than 80 % are empty -- triggered by wind or vegetation movement.
    MegaDetector is a pre-trained detector that filters those empty frames automatically,
    reducing manual review by ~80 %.

    MegaDetector detects three classes: **animal * person * vehicle**.
    It does *not* identify species -- species classification is a downstream problem (P5).

    We use the **agentmorris/MegaDetector** implementation -- the actively maintained
    Python package from the original author. In Part 2 you fine-tune a YOLOv8
    detector with Ultralytics on the bounding boxes you annotated in P2.

    **Install:**
    ```bash
    pip install megadetector   # MegaDetector inference (agentmorris fork)
    pip install yolov5         # YOLOv5 training -- required to fine-tune from MD weights
    ```

    > **Why `yolov5` and not `ultralytics`?**
    > MegaDetector v5 is a YOLOv5x6 model. The `ultralytics` package (YOLOv8/v11)
    > cannot load YOLOv5 checkpoints -- the architectures are incompatible.
    > Fine-tuning from the original MD weights requires the `yolov5` package.
    """)
    return


# ===============================================================================
# PART 1 -- Inference with MegaDetector
# ===============================================================================

@app.cell(hide_code=True)
def _part1(mo):
    mo.md("## Part 1 -- Inference with MegaDetector")
    return


@app.cell(hide_code=True)
def _step1(mo):
    mo.md(r"""
    ### Step 1 -- Load MegaDetector v5

    `load_detector('MDV5A')` downloads the ~600 MB model weights on first use
    and caches them locally. MDV5A is a YOLOv5-based model trained on
    ~4.5 million camera trap images from 89 data sources worldwide.

    The detector is loaded **once here** and reused in all later cells --
    Marimo's reactive graph ensures downstream cells re-run automatically
    if this cell changes, without re-loading the model unnecessarily.
    """)
    return


@app.cell
def _load_model():
    from megadetector.detection.run_detector import load_detector

    detector = load_detector("MDV5A")
    print(f"MegaDetector v5a loaded -- {type(detector).__name__}")
    return (detector,)


# -- Step 2: single image ------------------------------------------------------

@app.cell(hide_code=True)
def _step2(mo):
    mo.md(r"""
    ### Step 2 -- Single-image detection

    `generate_detections_one_image` returns a dict with:
    - `detections` -- list of hits, each with `category`, `conf`, `bbox`
    - `bbox` -- `[x_min, y_min, width, height]` **normalised 0-1**
    - `category` -- string: `'1'` = animal, `'2'` = person, `'3'` = vehicle

    The `detector` object comes from Step 1 -- no model re-load here.
    """)
    return


@app.cell
def _single_image(DATA_DIR, Image, detector, mpatches, np, plt):
    _LABELS = {"1": "animal", "2": "person", "3": "vehicle"}
    _COLORS = {"1": "#E74C3C", "2": "#3498DB", "3": "#F39C12"}

    _img_dir = DATA_DIR / "camera_trap" / "caltech_subset"
    _img_files = sorted(
        p for p in _img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}
    )

    if not _img_files:
        print("No Caltech images found -- run the data download step first.")
    else:
        _img_path = _img_files[0]
        _pil = Image.open(_img_path).convert("RGB")
        _arr = np.array(_pil)
        _W, _H = _pil.width, _pil.height

        _result = detector.generate_detections_one_image(
            _arr,
            image_id=_img_path.name,
            detection_threshold=0.1,
        )

        print(f"Image : {_img_path.name}  ({_W}x{_H} px)")
        print(f"Hits  : {len(_result['detections'])}")
        for _d in _result["detections"]:
            _lbl = _LABELS.get(_d["category"], "?")
            _bb = [round(v, 3) for v in _d["bbox"]]
            print(f"  {_lbl:8s}  conf={_d['conf']:.3f}  bbox={_bb}")

        _fig, _ax = plt.subplots(figsize=(10, 7))
        _ax.imshow(_arr)
        for _d in _result["detections"]:
            _x, _y, _bw, _bh = _d["bbox"]
            _col = _COLORS.get(_d["category"], "white")
            _lbl = _LABELS.get(_d["category"], "?")
            _ax.add_patch(mpatches.Rectangle(
                (_x * _W, _y * _H), _bw * _W, _bh * _H,
                linewidth=2, edgecolor=_col, facecolor="none",
            ))
            _ax.text(
                _x * _W, _y * _H - 4,
                f"{_lbl} {_d['conf']:.2f}",
                fontsize=8, color=_col, fontweight="bold",
            )
        _ax.set_title(f"MegaDetector v5a -- {_img_path.name}", fontsize=10)
        _ax.axis("off")
        plt.tight_layout()

    plt.gca()
    return


# -- Step 3: batch detection ---------------------------------------------------

@app.cell(hide_code=True)
def _step3(mo):
    mo.md(r"""
    ### Step 3 -- Batch detection

    `run_detector_batch` processes all images in a single vectorised pass --
    significantly faster than calling `generate_detections_one_image` in a
    Python loop because images are batched on-GPU and I/O is overlapped.
    It also supports checkpointing, so a crash mid-run is resumable.

    Results flow into a DataFrame used by all downstream steps:

    ```
    images  ->  MegaDetector  ->  animal crops  ->  species classifier (P5)
    ```
    """)
    return


@app.cell
def _batch_run(DATA_DIR, detector, pd):
    from megadetector.detection.run_detector_batch import load_and_run_detector_batch

    _CONF_THRESHOLD = 0.2
    _LABELS = {"1": "animal", "2": "person", "3": "vehicle"}

    _sources = {
        "caltech":   DATA_DIR / "camera_trap" / "caltech_subset",
        "serengeti": DATA_DIR / "camera_trap" / "serengeti_subset",
    }

    # Gather all image paths, tagging each with its source dataset
    _image_paths, _source_tags = [], {}
    for _source, _img_dir in _sources.items():
        if not _img_dir.exists():
            continue
        for _p in sorted(_img_dir.iterdir()):
            if _p.suffix.lower() in {".jpg", ".jpeg"}:
                _image_paths.append(str(_p))
                _source_tags[str(_p)] = _source

    # Single vectorised batch call -- much faster than a per-image loop.
    # Pass the already-loaded detector object to avoid a second model load.
    _batch_results = load_and_run_detector_batch(
        detector,
        _image_paths,
        confidence_threshold=_CONF_THRESHOLD,
        quiet=True,
    )

    # Flatten into a DataFrame
    _records = []
    for _img_result in _batch_results["images"]:
        _fp = _img_result["file"]
        for _d in _img_result.get("detections", []):
            _records.append({
                "source":     _source_tags.get(_fp, "unknown"),
                "filename":   Path(_fp).name,
                "filepath":   _fp,
                "category":   _LABELS.get(_d["category"], "unknown"),
                "confidence": round(_d["conf"], 4),
                "bbox_x":     round(_d["bbox"][0], 4),
                "bbox_y":     round(_d["bbox"][1], 4),
                "bbox_w":     round(_d["bbox"][2], 4),
                "bbox_h":     round(_d["bbox"][3], 4),
            })

    detections_df = pd.DataFrame(_records)

    _out = DATA_DIR / "camera_trap_detections.csv"
    detections_df.to_csv(_out, index=False)

    print(f"Images processed : {detections_df['filename'].nunique()}")
    print(f"Total detections : {len(detections_df)}")
    print()
    print(detections_df.groupby(["source", "category"]).size().to_string())
    print(f"\nSaved -> {_out}")

    return (detections_df,)


# -- Step 4: confidence threshold ----------------------------------------------

@app.cell(hide_code=True)
def _step4(mo):
    mo.md(r"""
    ### Step 4 -- Confidence threshold and empty-frame filtering

    The confidence threshold is the most important tuning parameter.
    Too low -> many false positives (wind, leaves). Too high -> missed animals.

    A common field deployment strategy: set threshold at 0.2, send all
    detections above it to a reviewer, skip everything below.
    """)
    return


@app.cell
def _threshold_analysis(detections_df, np, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))

    _axes[0].hist(
        detections_df["confidence"], bins=30,
        color="steelblue", edgecolor="white",
    )
    _axes[0].set_xlabel("Confidence score")
    _axes[0].set_ylabel("Number of detections")
    _axes[0].set_title("Confidence distribution -- all detections")
    for _t in [0.2, 0.5, 0.8]:
        _axes[0].axvline(_t, color="red", linestyle="--", linewidth=1,
                         alpha=0.6, label=f"t={_t}")
    _axes[0].legend(fontsize=8)

    _thresholds = np.arange(0.05, 0.99, 0.05)
    _counts = [
        len(detections_df[detections_df["confidence"] >= t])
        for t in _thresholds
    ]
    _axes[1].plot(_thresholds, _counts, "o-", color="darkorange")
    _axes[1].set_xlabel("Confidence threshold")
    _axes[1].set_ylabel("Detections retained")
    _axes[1].set_title("Detections retained vs. threshold")
    _axes[1].grid(alpha=0.3)

    plt.suptitle("Choosing a confidence threshold", fontsize=12)
    plt.tight_layout()
    plt.gca()
    return


# -- Step 5: crop extraction ---------------------------------------------------

@app.cell(hide_code=True)
def _step5(mo):
    mo.md(r"""
    ### Step 5 -- Extract animal crops

    Save each animal detection as a JPEG crop.
    These crops are the input to the species classifier in P5.
    """)
    return


@app.cell
def _crops(DATA_DIR, Image, detections_df):
    CROPS_DIR = DATA_DIR / "camera_trap_crops"
    CROPS_DIR.mkdir(exist_ok=True)

    _animals = detections_df[
        (detections_df["category"] == "animal") &
        (detections_df["confidence"] >= 0.2)
    ]

    _saved = 0
    for _i, _row in _animals.iterrows():
        _img = Image.open(_row["filepath"]).convert("RGB")
        _W, _H = _img.size
        _x1 = max(0, int(_row["bbox_x"] * _W))
        _y1 = max(0, int(_row["bbox_y"] * _H))
        _x2 = min(_W, int((_row["bbox_x"] + _row["bbox_w"]) * _W))
        _y2 = min(_H, int((_row["bbox_y"] + _row["bbox_h"]) * _H))
        _crop = _img.crop((_x1, _y1, _x2, _y2))
        _stem = _row["filename"].rsplit(".", 1)[0]
        _crop.save(
            CROPS_DIR / f"{_row['source']}_{_stem}_crop{_saved:04d}.jpg",
            quality=90,
        )
        _saved += 1

    print(f"Saved {_saved} animal crops -> {CROPS_DIR}")
    return (CROPS_DIR,)


# ===============================================================================
# PART 2 -- Fine-tuning a Custom Detector with Ultralytics
# ===============================================================================

@app.cell(hide_code=True)
def _part2(mo):
    mo.md(r"""
    ## Part 2 -- Fine-tuning a Custom Detector with Ultralytics

    MegaDetector was trained on millions of camera trap images from around the world.
    But what if your images come from a new context -- drone aerial view, unusual species,
    or a setting very different from the training data?

    In Part 2 you fine-tune a **YOLOv5x6** detector on the bounding box annotations
    you created in P2, starting from the **MegaDetector v5a weights** rather than
    generic COCO-pretrained weights. This gives the model a far better starting point:
    MegaDetector has already learned to find animals in camera trap images, so your
    fine-tuning only needs to adapt it to your specific context (new species, drone
    perspective, unusual habitat).

    ```
    Label Studio export (COCO JSON)
        -> convert to YOLO .txt format
        -> create dataset.yaml
        -> yolo train  (Ultralytics Trainer)
        -> evaluate on val set
    ```

    **Note:** If you haven't annotated enough images in P2 yet, the Caltech
    bounding boxes from `camera_trap_labels.csv` are used automatically as a fallback.
    """)
    return


# -- Step 6: convert annotations -----------------------------------------------

@app.cell(hide_code=True)
def _step6(mo):
    mo.md(r"""
    ### Step 6 -- Convert annotations to YOLO format

    YOLO expects one `.txt` file per image with lines:

    ```
    class_id  cx  cy  w  h     # all values normalised 0-1
    0  0.512  0.344  0.231  0.189
    ```

    We convert from either:
    - **COCO JSON** (Label Studio export): pixel `[x, y, w, h]` -> normalised centre
    - **Caltech CSV fallback**: same conversion from `bbox_x/y/w/h` columns

    This cell only builds the annotation dictionary -- no files are written yet,
    so re-running it is cheap and doesn't touch the filesystem.
    """)
    return


@app.cell
def _convert_annotations(DATA_DIR, Image):
    import json as _json

    CLASSES = ["animal", "person", "vehicle"]

    def _to_yolo_line(x, y, w, h, img_w, img_h, class_id):
        """Convert pixel COCO bbox to normalised YOLO centre format."""
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        return f"{class_id} {cx:.6f} {cy:.6f} {w / img_w:.6f} {h / img_h:.6f}"

    # annotations_by_file: filename -> list of YOLO label lines
    annotations_by_file: dict[str, list[str]] = {}

    # -- Source 1: Label Studio COCO JSON from P2 -----------------------------
    _ls_path = DATA_DIR / "my_serengeti_bboxes.json"
    if _ls_path.exists():
        print(f"Loading Label Studio export: {_ls_path.name}")
        with open(_ls_path) as _f:
            _coco = _json.load(_f)
        _id_to_img = {img["id"]: img for img in _coco["images"]}
        _id_to_cat = {c["id"]: c["name"].lower() for c in _coco["categories"]}
        for _ann in _coco["annotations"]:
            if not _ann.get("bbox"):
                continue
            _img  = _id_to_img[_ann["image_id"]]
            _fname = _img["file_name"].split("/")[-1]
            _cls  = CLASSES.index(_id_to_cat.get(_ann["category_id"], "animal"))
            _x, _y, _w, _h = _ann["bbox"]
            annotations_by_file.setdefault(_fname, []).append(
                _to_yolo_line(_x, _y, _w, _h, _img["width"], _img["height"], _cls)
            )
        print(f"  {len(annotations_by_file)} annotated images")

    # -- Source 2: Caltech CSV fallback ----------------------------------------
    _caltech_csv = DATA_DIR / "camera_trap_labels.csv"
    _caltech_dir = DATA_DIR / "camera_trap" / "caltech_subset"
    if not annotations_by_file and _caltech_csv.exists():
        print("No Label Studio export found -- using Caltech CSV fallback.")
        import pandas as _pd
        _df = _pd.read_csv(_caltech_csv).dropna(subset=["bbox_x"])
        for _, _row in _df.iterrows():
            _src = _caltech_dir / _row["crop"]
            if not _src.exists():
                continue
            with Image.open(_src) as _im:
                _iw, _ih = _im.size
            annotations_by_file.setdefault(_row["crop"], []).append(
                _to_yolo_line(
                    _row["bbox_x"], _row["bbox_y"],
                    _row["bbox_w"], _row["bbox_h"],
                    _iw, _ih, 0,   # class 0 = animal
                )
            )
        print(f"  {len(annotations_by_file)} annotated images (Caltech fallback)")

    print(f"\nTotal annotations: {sum(len(v) for v in annotations_by_file.values())}")

    return CLASSES, annotations_by_file


# -- Step 7: build YOLO dataset on disk ----------------------------------------

@app.cell(hide_code=True)
def _step7_md(mo):
    mo.md(r"""
    ### Step 7 -- Write YOLO dataset to disk

    Split annotated images 80/20 into train and val, copy images,
    write label `.txt` files, and generate `dataset.yaml`.

    Kept separate from Step 6 so the annotation conversion is cheap to re-run
    without triggering a full filesystem write.
    """)
    return


@app.cell
def _build_dataset(CLASSES, DATA_DIR, annotations_by_file):
    import shutil as _shutil

    YOLO_DIR = DATA_DIR / "yolo_dataset"

    # Create directory tree
    for _split in ("train", "val"):
        for _sub in ("images", "labels"):
            (YOLO_DIR / _sub / _split).mkdir(parents=True, exist_ok=True)

    # Source image directories (searched in order)
    _src_dirs = [
        DATA_DIR / "camera_trap" / "serengeti_subset",
        DATA_DIR / "camera_trap" / "caltech_subset",
        DATA_DIR / "ls_serengeti",
        DATA_DIR / "ls_caltech",
    ]

    def _find_image(filename):
        for _d in _src_dirs:
            _p = _d / filename
            if _p.exists():
                return _p
        return None

    # 80/20 split
    _fnames = sorted(annotations_by_file)
    _cut = max(1, int(len(_fnames) * 0.8))
    _splits = {"train": _fnames[:_cut], "val": _fnames[_cut:]}

    _counts = {}
    for _split, _names in _splits.items():
        _written = 0
        for _fn in _names:
            _src = _find_image(_fn)
            if _src is None:
                continue
            _shutil.copy(_src, YOLO_DIR / "images" / _split / _fn)
            _stem = _src.stem
            (YOLO_DIR / "labels" / _split / f"{_stem}.txt").write_text(
                "\n".join(annotations_by_file[_fn])
            )
            _written += 1
        _counts[_split] = _written

    # Write dataset.yaml
    _yaml_text = (
        f"path: {YOLO_DIR}\n"
        f"train: images/train\n"
        f"val:   images/val\n\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )
    (YOLO_DIR / "dataset.yaml").write_text(_yaml_text)

    print(f"YOLO dataset ready:")
    print(f"  train : {_counts['train']} images")
    print(f"  val   : {_counts['val']}   images")
    print(f"  yaml  : {YOLO_DIR / 'dataset.yaml'}")

    return (YOLO_DIR,)


# -- Step 8: train -------------------------------------------------------------

@app.cell(hide_code=True)
def _step8_md(mo):
    mo.md(r"""
    ### Step 8 -- Fine-tune from MegaDetector v5a weights

    We start from the **MegaDetector v5a checkpoint** (`md_v5a.0.0.pt`, ~600 MB)
    rather than generic COCO-pretrained weights. This matters: MD has already learned
    to find animals in camera trap images across 89 global datasets, so fine-tuning
    adapts that knowledge rather than relearning it from scratch.

    Training uses the `yolov5` package (not `ultralytics`) because MDv5a is a
    **YOLOv5x6** model -- a different architecture that YOLOv8/v11 cannot load.

    Key hyperparameters:
    - `imgsz=1280` -- match MDv5's original training resolution (important for small animals)
    - `freeze=10` -- freeze the first 10 backbone layers; only the neck and head adapt.
      This prevents catastrophic forgetting of MD's animal-detection features.
      Set `freeze=0` to unfreeze everything if you have >=500 annotated images.
    - `batch=4` -- 1280 px images are large; reduce to 2 if you hit OOM
    - `epochs=50` -- enough to converge on small datasets; use 100-300 for production

    The weights file is downloaded once and cached in `DATA_DIR`.
    Training results land in `YOLO_DIR/runs/train_md_finetune/`.
    """)
    return


@app.cell
def _train(DATA_DIR, YOLO_DIR, torch):
    import urllib.request as _urllib_req
    import yolov5 as _yv5

    # -- Download MD weights if not already cached ------------------------------
    # The megadetector package caches weights in a runtime temp dir that doesn't
    # persist across sessions; we keep a stable copy in DATA_DIR instead.
    _MD_URL = (
        "https://github.com/agentmorris/MegaDetector/releases/download/v5.0/"
        "md_v5a.0.0.pt"
    )
    _md_weights = DATA_DIR / "md_v5a.0.0.pt"
    if not _md_weights.exists():
        print("Downloading MegaDetector v5a weights (~600 MB) ...")
        _urllib_req.urlretrieve(_MD_URL, _md_weights)
        print("Download complete.")
    else:
        print(f"Using cached MD weights: {_md_weights.name}")

    # -- Device ----------------------------------------------------------------
    _device = "0" if torch.cuda.is_available() else "cpu"
    _gpu_name = torch.cuda.get_device_name(0) if _device == "0" else "CPU"
    print(f"Training on: {_gpu_name}")

    # -- Train -----------------------------------------------------------------
    # yolov5.train.run() is the programmatic equivalent of:
    #   python -m yolov5.train --weights md_v5a.0.0.pt --data dataset.yaml ...
    _train_dir = YOLO_DIR / "runs" / "train_md_finetune"

    _yv5.train.run(
        weights=str(_md_weights),
        data=str(YOLO_DIR / "dataset.yaml"),
        epochs=50,
        imgsz=1280,       # match MDv5 training resolution
        batch=4,          # 1280 px is large; reduce to 2 if OOM
        device=_device,
        project=str(YOLO_DIR / "runs"),
        name="train_md_finetune",
        exist_ok=True,
        # Freeze the first 10 backbone layers -- preserves MD's learned features.
        # Set freeze=0 to unfreeze everything once you have >=500 annotations.
        freeze=10,
        # Augmentation -- conservative for fine-tuning a pretrained model
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        fliplr=0.5,
        mosaic=0.5,       # lower than default; MD already saw mosaic at training
        mixup=0.0,
        # Artefacts
        save_period=10,   # checkpoint every 10 epochs
        plots=True,       # PR curve, confusion matrix, etc.
    )

    _best = _train_dir / "weights" / "best.pt"
    print(f"\nTraining complete.")
    print(f"  Best weights : {_best}")

    # Load the fine-tuned model for the evaluation cell below
    model = _yv5.load(str(_best))
    model.conf = 0.2      # confidence threshold for predictions

    return (model,)


# -- Step 9: evaluate on validation images ------------------------------------

@app.cell(hide_code=True)
def _step9_md(mo):
    mo.md(r"""
    ### Step 9 -- Run the custom detector on validation images

    Compare your trained detector against MegaDetector on the same images.
    At this annotation scale (tens of images) the custom detector will likely
    underperform MegaDetector -- but you can see it learning, and mAP should
    improve as you add more annotations from P2.
    """)
    return


@app.cell
def _evaluate(YOLO_DIR, Image, mpatches, model, np, plt):
    # yolov5 API: model(img_array) -> Results object
    # results.xyxy[0] is a (N, 6) tensor: [x1, y1, x2, y2, conf, class_id]
    _COLORS = ["#E74C3C", "#3498DB", "#F39C12"]

    _val_dir = YOLO_DIR / "images" / "val"
    _val_imgs = sorted(_val_dir.iterdir())[:6]

    if not _val_imgs:
        print("No validation images found.")
    else:
        _ncols = 3
        _nrows = -(-len(_val_imgs) // _ncols)   # ceiling division
        _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(_ncols * 5, _nrows * 4))
        _axes_flat = np.array(_axes).flatten()

        for _i, _p in enumerate(_val_imgs):
            _arr = np.array(Image.open(_p).convert("RGB"))
            _results = model(_arr)                 # yolov5 inference
            _preds   = _results.xyxy[0].numpy()   # shape (N, 6)

            _ax = _axes_flat[_i]
            _ax.imshow(_arr)
            for _row in _preds:
                _x1, _y1, _x2, _y2, _conf, _cls = _row
                _col = _COLORS[int(_cls) % len(_COLORS)]
                _ax.add_patch(mpatches.Rectangle(
                    (_x1, _y1), _x2 - _x1, _y2 - _y1,
                    linewidth=2, edgecolor=_col, facecolor="none",
                ))
                _ax.text(_x1, _y1 - 4, f"{_conf:.2f}", fontsize=7, color=_col)
            _ax.set_title(_p.name, fontsize=7)
            _ax.axis("off")

        for _ax in _axes_flat[len(_val_imgs):]:
            _ax.axis("off")

        plt.suptitle("Fine-tuned MegaDetector (YOLOv5x6) -- validation images", fontsize=11)
        plt.tight_layout()

    plt.gca()
    return


# ===============================================================================
# Exercises and Reflection
# ===============================================================================

@app.cell(hide_code=True)
def _exercise(mo):
    mo.md(r"""
    ## Exercise

    1. **Threshold sweep** -- Change `_CONF_THRESHOLD` in Step 3 to 0.05, 0.2, and 0.5.
       At each threshold, how many images are classified as "empty" (zero detections)?
       What is the false-positive rate at 0.05?

    2. **Failure modes** -- Browse your crops in `data/camera_trap_crops/`.
       Find one of each:
       - A confident correct detection (conf > 0.8)
       - A confident false positive (conf > 0.5 but clearly not an animal)
       - A missed animal (visible in the original image, no detection)

    3. **Training data size** -- Your custom model was trained on very few images.
       After adding more annotations from P2, re-run Steps 6-8. Does mAP improve?
       How many images do you need before it approaches MegaDetector performance?

    4. **Frozen vs. unfrozen backbone** -- Re-run Step 8 with `freeze=0` (full fine-tune)
       and `freeze=24` (freeze everything up to the detection head).
       Compare mAP50 across the three settings. When does freezing help?
       (Hint: think about dataset size.)

    5. **Resolution matters** -- Change `imgsz` from 1280 to 640 in Step 8.
       How does training time change? Does mAP change?
       When might 640 be preferable despite the accuracy trade-off?

    6. **MegaDetector on aerial imagery** -- Run MegaDetector on one HerdNet tile
       (from `data/general_dataset/test_sample/`). What does it detect?
       Is a camera-trap detector appropriate for drone data?
    """)
    return


@app.cell(hide_code=True)
def _reflection(mo):
    mo.md(r"""
    ## Reflection

    - MegaDetector was trained on 4.5 million images from 89 sources.
      Your custom detector was trained on tens of images.
      What does this say about the role of pre-trained foundation models in ecology?

    - MegaDetector detects *animal* as one class -- it does not distinguish
      zebra from elephant. What are the implications for a species monitoring pipeline?

    - You used citizen-science species labels (Serengeti) to draw bounding boxes.
      If those species labels are wrong, how does that affect:
      (a) the bounding box quality?
      (b) the downstream classifier trained on those crops?

    - YOLOv8s achieves competitive accuracy on standard benchmarks at ~22 MB.
      MegaDetector v5 is ~600 MB. When would you prefer the smaller model?

    - You fine-tuned from MegaDetector weights rather than COCO-pretrained weights.
      What are the trade-offs? When might COCO weights be a better starting point?

    - The `freeze=10` setting prevents the first 10 backbone layers from updating.
      Why might updating those layers be harmful when you have very few annotations?
      What would happen to a model trained from scratch on the same tiny dataset?
    """)
    return


if __name__ == "__main__":
    app.run()
