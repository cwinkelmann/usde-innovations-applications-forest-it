import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _context(mo):
    mo.md(r"""
    # Practical 7 — Introduction to Segmentation

    **Context:** Detection tells you *where* an animal is (a box or point).
    Segmentation tells you *which pixels* belong to each class. This matters for:
    - **Habitat mapping** — which substrate type does each pixel belong to?
    - **Land cover change detection** — is this patch of forest now bare ground?
    - **Week 2 bridge** — next week you will apply these ideas to SAR imagery
      for detecting illegal mining in Ghanaian Biosphere Reserves.

    Today you will:
    - Run SAM (Segment Anything Model) on a drone image using point prompts
    - Understand the difference between semantic and instance segmentation
    - Apply a simple pixel classifier to a land cover scene
    - See how segmentation outputs connect to the detection workflow

    **Install:** `pip install segment-anything`
    Download SAM weights: `vit_b` checkpoint from Meta (~375 MB)
    """)


@app.cell
def _imports():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from PIL import Image

    return Image, Path, np, plt, torch


@app.cell
def _step1(mo):
    mo.md(r"""
    ## Step 1 — Load SAM

    SAM (Kirillov et al., 2023) is a foundation model for segmentation. It takes
    an image plus a *prompt* (point, box, or text) and returns a binary mask.

    Three model sizes: `vit_b` (fast), `vit_l` (balanced), `vit_h` (best quality).
    We use `vit_b` for speed in this practical.
    """)


@app.cell
def _load_sam(Path, torch):
    from segment_anything import sam_model_registry, SamPredictor

    CHECKPOINT = Path("../data/sam_vit_b_01ec64.pth")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if not CHECKPOINT.exists():
        print(f"SAM checkpoint not found at {CHECKPOINT}")
        print("Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        sam = None
        predictor = None
    else:
        sam = sam_model_registry["vit_b"](checkpoint=str(CHECKPOINT))
        sam.to(device=DEVICE)
        predictor = SamPredictor(sam)
        print(f"SAM loaded on {DEVICE}")

    return CHECKPOINT, DEVICE, SamPredictor, predictor, sam, sam_model_registry


@app.cell
def _step2(mo):
    mo.md(r"""
    ## Step 2 — Segment with a point prompt

    SAM takes a point (x, y) and a label (1 = foreground, 0 = background).
    It returns three candidate masks with confidence scores — you pick the best.

    Click on an animal or habitat feature in the image below, then run the cell
    with those coordinates.
    """)


@app.cell
def _point_segmentation(Image, Path, np, plt, predictor):
    _IMAGE_PATH = Path("../data/sample_tile.jpg")

    if predictor is None or not _IMAGE_PATH.exists():
        print("SAM not loaded or image not found. Skipping.")
    else:
        image = np.array(Image.open(_IMAGE_PATH).convert("RGB"))
        predictor.set_image(image)

        # Prompt: click on a point of interest (x, y in pixel coords)
        INPUT_POINT = np.array([[256, 256]])  # ← change this to a point on an animal
        INPUT_LABEL = np.array([1])           # 1 = foreground

        masks, scores, logits = predictor.predict(
            point_coords=INPUT_POINT,
            point_labels=INPUT_LABEL,
            multimask_output=True,
        )

        # Plot the three candidate masks
        _fig, _axes = plt.subplots(1, 4, figsize=(16, 4))

        _axes[0].imshow(image)
        _axes[0].plot(*INPUT_POINT[0], "r*", markersize=12)
        _axes[0].set_title("Input image + prompt")
        _axes[0].axis("off")

        for i, (mask, score) in enumerate(zip(masks, scores)):
            _axes[i + 1].imshow(image)
            _axes[i + 1].imshow(mask, alpha=0.5, cmap="Reds")
            _axes[i + 1].set_title(f"Mask {i + 1}  (score={score:.3f})")
            _axes[i + 1].axis("off")

        plt.suptitle("SAM point-prompted segmentation — three candidate masks", fontsize=12)
        plt.tight_layout()
        plt.show()
    return INPUT_LABEL, INPUT_POINT, image, logits, mask, masks, score, scores


@app.cell
def _step3(mo):
    mo.md(r"""
    ## Step 3 — Semantic segmentation with a simple pixel classifier

    SAM is instance-level (one mask per prompt). For habitat mapping, we want
    *semantic* segmentation: classify every pixel into a land cover class.

    A simple baseline: extract colour features per pixel and train a random forest.
    This is not deep learning — it is deliberately simple to show the concept.
    """)


@app.cell
def _semantic_seg(Image, Path, np, plt):
    from sklearn.ensemble import RandomForestClassifier

    _IMAGE_PATH2 = Path("../data/sample_tile.jpg")
    LABELS_PATH = Path("../data/sample_tile_labels.npy")  # (H, W) uint8 mask with class indices

    CLASS_NAMES = ["rock", "vegetation", "sand", "water", "iguana"]
    COLOURS = ["grey", "green", "yellow", "blue", "red"]

    if not _IMAGE_PATH2.exists() or not LABELS_PATH.exists():
        print("Image or labels not found — using synthetic demo data.")
        H, W = 256, 256
        rng = np.random.default_rng(42)
        image_arr = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
        labels_arr = rng.integers(0, len(CLASS_NAMES), (H, W), dtype=np.uint8)
    else:
        image_arr = np.array(Image.open(_IMAGE_PATH2).convert("RGB"))
        labels_arr = np.load(str(LABELS_PATH))
        H, W = image_arr.shape[:2]

    # Feature matrix: each pixel → [R, G, B]
    X = image_arr.reshape(-1, 3).astype(float) / 255.0
    y = labels_arr.flatten()

    # Only use labelled pixels (label < 255 = labelled; 255 = unlabelled/ignored)
    labelled = y < len(CLASS_NAMES)
    X_labelled = X[labelled]
    y_labelled = y[labelled]

    # Train a random forest on 80 % of labelled pixels
    split = int(0.8 * len(X_labelled))
    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf.fit(X_labelled[:split], y_labelled[:split])

    acc = clf.score(X_labelled[split:], y_labelled[split:])
    print(f"Pixel-level accuracy on validation set: {acc:.2%}")

    # Predict all pixels
    y_pred = clf.predict(X)
    pred_map = y_pred.reshape(H, W)

    # Visualise
    import matplotlib.colors as mcolors
    cmap = mcolors.ListedColormap(COLOURS[:len(CLASS_NAMES)])

    _fig, _axes = plt.subplots(1, 3, figsize=(14, 5))
    _axes[0].imshow(image_arr)
    _axes[0].set_title("Input tile")
    _axes[0].axis("off")

    _axes[1].imshow(labels_arr, cmap=cmap, vmin=0, vmax=len(CLASS_NAMES) - 1)
    _axes[1].set_title("Ground truth labels")
    _axes[1].axis("off")

    im = _axes[2].imshow(pred_map, cmap=cmap, vmin=0, vmax=len(CLASS_NAMES) - 1)
    _axes[2].set_title(f"Predicted land cover (acc={acc:.1%})")
    _axes[2].axis("off")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=n) for c, n in zip(COLOURS, CLASS_NAMES)]
    _fig.legend(handles=legend_elements, loc="lower center", ncol=len(CLASS_NAMES), fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.show()
    return (
        CLASS_NAMES, COLOURS, LABELS_PATH, Patch, RandomForestClassifier, X, X_labelled,
        acc, axes, clf, cmap, fig, im, image_arr, labelled, labels_arr, legend_elements,
        mcolors, pred_map, split, y, y_labelled, y_pred,
    )


@app.cell
def _exercise(mo):
    mo.md(r"""
    ## Exercise

    > **Change `INPUT_POINT` in Step 2 to click on different features — a rock,
    > a patch of vegetation, the water's edge. Does SAM return sensible masks?**

    For Step 3, try adding a fourth feature channel:
    ```python
    # Compute a simple vegetation index (greenness)
    green_excess = image_arr[:,:,1].astype(float) - image_arr[:,:,0].astype(float)
    X = np.stack([...original channels..., green_excess.flatten()], axis=1)
    ```
    Does adding this feature improve accuracy?

    **Bridge to Week 2:** SAR imagery does not have RGB bands — it has backscatter
    intensity (and sometimes phase). The same pixel-classifier approach still works,
    but with different features (texture, ratio of VV/VH polarisations).
    """)


@app.cell
def _reflection(mo):
    mo.md(r"""
    ## Reflection

    - What is the difference between SAM (instance segmentation) and a random forest
      pixel classifier (semantic segmentation)?
    - The random forest achieved high accuracy on colourful test data. Would it
      generalise to a different island with different rock colours?
    - SAM needs no training data — it uses a foundation model. What are the limits
      of that approach for specialised remote sensing tasks?

    **Bridge:** In Week 2, you will apply these segmentation ideas to Sentinel-1 SAR
    imagery, where the input bands are radar backscatter intensities rather than RGB.
    The concept is the same; the features are different.
    """)


if __name__ == "__main__":
    app.run()
