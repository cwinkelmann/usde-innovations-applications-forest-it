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
    # Practical 5 — Species Classification with a Pre-Trained Model

    **Context:** MegaDetector tells you *there is an animal*. To know *which species*,
    you pass the crop to a classifier. This is the second stage of the two-stage pipeline.

    We use `timm` (PyTorch Image Models) — a library of hundreds of pre-trained models
    that can be used for inference or fine-tuned on your own data.

    Today you will:
    - Load a pre-trained EfficientNet from `timm`
    - Run inference on the animal crops from Practical 3
    - Build a results table: crop → predicted class → confidence
    - Understand what the model was trained on and where it will fail

    **Install:** `pip install timm torch torchvision`
    """)


@app.cell
def _imports():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import timm
    from PIL import Image

    return Image, Path, np, pd, plt, timm, torch


@app.cell
def _step1(mo):
    mo.md(r"""
    ## Step 1 — Load a pre-trained EfficientNet

    `timm.create_model` downloads weights from HuggingFace on first use.
    `pretrained=True` loads ImageNet-1k weights — 1000 classes including many animals.

    We set `model.eval()` to disable dropout and batch normalisation training behaviour.
    """)


@app.cell
def _load_model(timm, torch):
    MODEL_NAME = "efficientnet_b0"  # small and fast; try "efficientnet_b3" for more accuracy

    model = timm.create_model(MODEL_NAME, pretrained=True)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model      : {MODEL_NAME}")
    print(f"Parameters : {n_params:,}")
    print(f"Output classes: {model.num_classes}")
    return MODEL_NAME, model


@app.cell
def _step2(mo):
    mo.md(r"""
    ## Step 2 — Build the preprocessing pipeline

    The model expects images normalised with ImageNet statistics. `timm` provides
    `data_config` and `create_transform` to build the correct preprocessing for any model.
    """)


@app.cell
def _preprocessing(model, timm):
    from timm.data import resolve_data_config, create_transform

    data_config = resolve_data_config({}, model=model)
    transform = create_transform(**data_config)

    print("Input size :", data_config["input_size"])
    print("Mean       :", data_config["mean"])
    print("Std        :", data_config["std"])
    return create_transform, data_config, resolve_data_config, transform


@app.cell
def _step3(mo):
    mo.md(r"""
    ## Step 3 — Load ImageNet class labels

    The model outputs a 1000-dimensional vector. We map the argmax to a human-readable
    label using the standard ImageNet class list.
    """)


@app.cell
def _load_labels():
    import urllib.request
    import json

    LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    with urllib.request.urlopen(LABELS_URL) as resp:
        imagenet_labels = json.load(resp)

    print(f"Loaded {len(imagenet_labels)} ImageNet class labels")
    print("Sample wildlife classes:", [l for l in imagenet_labels if "iguana" in l.lower() or "lizard" in l.lower()])
    return LABELS_URL, imagenet_labels, json


@app.cell
def _step4(mo):
    mo.md(r"""
    ## Step 4 — Run inference on animal crops

    For each crop: preprocess → forward pass → softmax → top-5 predictions.
    """)


@app.cell
def _run_inference(Image, Path, imagenet_labels, model, pd, torch, transform):
    CROPS_DIR = Path("../data/camera_trap_crops")
    TOP_K = 5

    crop_files = sorted(CROPS_DIR.glob("*.jpg"))[:20]  # limit to first 20 for speed
    records = []

    with torch.no_grad():
        for crop_path in crop_files:
            img = Image.open(crop_path).convert("RGB")
            tensor = transform(img).unsqueeze(0)  # (1, C, H, W)

            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

            top_indices = probs.topk(TOP_K).indices.tolist()
            top_probs = probs.topk(TOP_K).values.tolist()

            records.append({
                "crop": crop_path.name,
                "top1_label": imagenet_labels[top_indices[0]],
                "top1_conf": round(top_probs[0], 4),
                "top2_label": imagenet_labels[top_indices[1]],
                "top2_conf": round(top_probs[1], 4),
                "top3_label": imagenet_labels[top_indices[2]],
                "top3_conf": round(top_probs[2], 4),
            })

    results_df = pd.DataFrame(records)
    print(results_df[["crop", "top1_label", "top1_conf"]].to_string(index=False))
    return CROPS_DIR, TOP_K, crop_files, crop_path, img, logits, probs, records, results_df, tensor, top_indices, top_probs


@app.cell
def _step5(mo):
    mo.md(r"""
    ## Step 5 — Visualise predictions

    Display crops with their top-1 prediction and confidence score.
    Look for cases where the model is confidently wrong.
    """)


@app.cell
def _visualise(Image, crop_files, plt, results_df):
    N_SHOW = min(10, len(crop_files))
    fig, axes = plt.subplots(2, N_SHOW // 2, figsize=(N_SHOW * 2, 6))
    axes = axes.flatten()

    for i, (ax, row) in enumerate(zip(axes, results_df.head(N_SHOW).itertuples())):
        img = Image.open(Path("../data/camera_trap_crops") / row.crop)
        ax.imshow(img)
        colour = "green" if row.top1_conf > 0.6 else "orange" if row.top1_conf > 0.3 else "red"
        ax.set_title(
            f"{row.top1_label}\n{row.top1_conf:.2%}",
            fontsize=8,
            color=colour,
        )
        ax.axis("off")

    for ax in axes[N_SHOW:]:
        ax.axis("off")

    plt.suptitle("EfficientNet top-1 predictions on animal crops", fontsize=12)
    plt.tight_layout()
    plt.show()
    return N_SHOW, ax, axes, colour, fig, i, img, row


@app.cell
def _exercise(mo):
    mo.md(r"""
    ## Exercise

    > **Change `MODEL_NAME` to `"efficientnet_b3"` and re-run. Does accuracy improve?**

    Also try `"resnet50"` or `"vit_base_patch16_224"` (Vision Transformer — slower but
    often more accurate).

    Compare the top-1 predictions for 5 crops across the three models:

    | Crop | EfficientNet-B0 | EfficientNet-B3 | ResNet-50 |
    |------|----------------|----------------|-----------|
    | ... | ... | ... | ... |

    **Key question:** All three models are trained on ImageNet. Would any of them
    reliably classify Galápagos marine iguanas? Why or why not?
    """)


@app.cell
def _reflection(mo):
    mo.md(r"""
    ## Reflection

    - ImageNet contains "common iguana" and "green lizard" classes. Does that mean
      EfficientNet-B0 can identify Galápagos marine iguanas reliably?
    - What would you need to fine-tune the classifier on iguana species?
      (Hint: what data, how many images, what labels?)
    - If your classifier is 97 % accurate on ImageNet but your wildlife dataset has
      species not in ImageNet, what accuracy do you expect on your data?
    """)


if __name__ == "__main__":
    app.run()
