# Workflow Agent — Active Learning Wildlife

## Role

The workflow agent orchestrates the full active learning loop for wildlife detection
using HILDA. It manages the end-to-end cycle from initialization through repeated
rounds of prediction, sample selection, annotation, and retraining.

## The Active Learning Loop

### Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ACTIVE LEARNING LOOP                         │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │1.Initialize│──▶│2.Predict │──▶│3.Select  │──▶│4.Export  │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│                                                       │        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐        ▼        │
│  │9.Evaluate│◀──│8.Retrain │◀──│7.Merge   │    ┌──────────┐ │
│  └──────────┘    └──────────┘    └──────────┘    │5.Annotate│ │
│       │                               ▲          └──────────┘ │
│       ▼                               │               │        │
│  ┌──────────┐                         │          ┌──────────┐ │
│  │10.Repeat?│─── yes ──▶ step 2       │          │6.Download│ │
│  └──────────┘                         └──────────┘──────────┘ │
│       │ no                                                      │
│       ▼                                                         │
│    DONE                                                         │
└─────────────────────────────────────────────────────────────────┘
```

### Step 1: Initialize

Set up the active learning environment.

```python
import torch
from pathlib import Path
from hilda.sampling import RGBContrastSampler, EmbeddingClusterSampler, UncertaintySampler

# Determine device
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Load or initialize model
if config["model_path"]:
    model = load_model(config["model_path"], device=device)
else:
    model = None  # Cold start

# Collect unlabeled image paths
unlabeled_pool = list(Path(config["unlabeled_dir"]).glob("*.jpg"))
unlabeled_pool += list(Path(config["unlabeled_dir"]).glob("*.png"))

# Load existing annotations if available
if config["existing_annotations"]:
    train_annotations = load_coco_json(config["existing_annotations"])
    annotated_images = set(train_annotations.keys())
else:
    train_annotations = {}
    annotated_images = set()

# Remove already-annotated images from pool
unlabeled_pool = [p for p in unlabeled_pool if str(p) not in annotated_images]

# Initialize logging
learning_curve = []
```

### Step 2: Predict

Run the current model on the unlabeled pool to get predictions and uncertainty scores.

```python
def predict_on_pool(model, unlabeled_pool, device):
    """Run model inference on all unlabeled images."""
    predictions = {}
    for img_path in unlabeled_pool:
        pred = model.predict(img_path, device=device)
        predictions[str(img_path)] = {
            "boxes": pred.boxes,       # or "points" for point-based models
            "scores": pred.scores,
            "labels": pred.labels,
            "logits": pred.logits,     # raw logits for uncertainty
        }
    return predictions
```

**Cold start variant:** If no model exists, skip this step. The sampling strategy
(RGB Contrast or random) will operate without predictions.

### Step 3: Select

Apply the configured sampling strategy to pick the most informative images.

```python
def select_samples(config, unlabeled_pool, predictions=None, annotated=[]):
    """Select top-K most informative images."""
    n_samples = config["images_per_round"]
    strategy = config["sampling_strategy"]

    if strategy == "rgb_contrast":
        sampler = RGBContrastSampler(n_samples=n_samples)
        indices = sampler.select(unlabeled_pool, already_annotated=annotated)

    elif strategy == "embedding_clustering":
        sampler = EmbeddingClusterSampler(
            n_samples=n_samples,
            feature_extractor=config["strategy_params"].get("feature_extractor", "dinov2"),
            n_clusters=config["strategy_params"].get("n_clusters", n_samples // 5),
        )
        indices = sampler.select(unlabeled_pool)

    elif strategy == "logit_uncertainty":
        sampler = UncertaintySampler(
            n_samples=n_samples,
            model=model,
            uncertainty_metric=config["strategy_params"].get("uncertainty_metric", "entropy"),
        )
        indices = sampler.select(unlabeled_pool)

    selected_images = [unlabeled_pool[i] for i in indices]
    return selected_images
```

### Step 4: Export

Format selected images and model predictions for the annotation tool.

```python
def export_to_annotation_tool(selected_images, predictions, config):
    """Export selected images + pre-annotations to CVAT or Label Studio."""
    tool = config["annotation_tool"]

    if tool == "cvat":
        # See annotation_tool_agent for CVAT export details
        export_to_cvat(
            images=selected_images,
            predictions={str(p): predictions.get(str(p)) for p in selected_images},
            project_name=f"AL_round_{current_round}",
            cvat_url=config.get("cvat_url", "http://localhost:8080"),
        )
    elif tool == "label_studio":
        # See annotation_tool_agent for Label Studio export details
        export_to_label_studio(
            images=selected_images,
            predictions={str(p): predictions.get(str(p)) for p in selected_images},
            project_name=f"AL_round_{current_round}",
            ls_url=config.get("label_studio_url", "http://localhost:8081"),
        )

    print(f"Exported {len(selected_images)} images to {tool}")
    print("Waiting for expert annotation...")
```

### Step 5: Annotate (Human Step)

This is the **manual step** where the expert reviews and corrects the model's predictions.

**What the expert does:**
- Open the annotation tool (CVAT or Label Studio)
- Review each image with its pre-annotations (model predictions shown as initial labels)
- Correct errors: fix misplaced boxes/points, remove false positives, add missed detections
- Mark each image as reviewed/completed

**Pre-annotation benefits:**
- Expert corrects rather than annotates from scratch
- 3-5x faster than annotating blank images
- Model predictions serve as a starting point, reducing missed detections

**The workflow pauses here** until the expert signals completion.

### Step 6: Download

Pull corrected annotations from the annotation tool.

```python
def download_annotations(config, round_num):
    """Download corrected annotations from CVAT or Label Studio."""
    tool = config["annotation_tool"]
    export_format = config["export_format"]

    if tool == "cvat":
        annotations = download_from_cvat(
            project_name=f"AL_round_{round_num}",
            format=export_format,
        )
    elif tool == "label_studio":
        annotations = download_from_label_studio(
            project_name=f"AL_round_{round_num}",
            format=export_format,
        )

    return annotations
```

### Step 7: Merge

Combine new annotations with the existing training set.

```python
def merge_annotations(existing, new_annotations, output_path):
    """Merge new annotations into the existing training set."""
    merged = {**existing}

    for img_path, annotation in new_annotations.items():
        if img_path in merged:
            # Update existing annotation (expert corrected it)
            merged[img_path] = annotation
        else:
            # Add new annotation
            merged[img_path] = annotation

    save_coco_json(merged, output_path)
    print(f"Training set: {len(merged)} images ({len(new_annotations)} new)")
    return merged
```

### Step 8: Retrain

Train the model on the expanded dataset.

```python
def retrain_model(config, train_annotations, round_num):
    """Retrain the model on the expanded training set."""
    # See herdnet-training skill for full training details
    model = train(
        annotations=train_annotations,
        val_split=config["val_split"],
        model_name=config.get("model_name", "herdnet"),
        epochs=config.get("epochs_per_round", 50),
        checkpoint_dir=Path(config["output_dir"]) / f"round_{round_num}" / "checkpoints",
        device=config["device"],
    )
    return model
```

### Step 9: Evaluate

Check improvement on the held-out validation set.

```python
def evaluate_round(model, val_dataset, round_num, n_total_annotations, learning_curve):
    """Evaluate the model after this round and log results."""
    metrics = model.evaluate(val_dataset)

    result = {
        "round": round_num,
        "n_annotations": n_total_annotations,
        "val_f1": metrics["f1"],
        "val_precision": metrics["precision"],
        "val_recall": metrics["recall"],
        "val_mae": metrics.get("mae"),  # Mean absolute error for counting
    }
    learning_curve.append(result)

    print(f"Round {round_num}: F1={result['val_f1']:.3f}, "
          f"annotations={n_total_annotations}")

    return result
```

### Step 10: Repeat or Stop

Decide whether to continue the loop.

```python
def should_continue(config, learning_curve, current_round, total_annotated):
    """Determine if the active learning loop should continue."""
    # Check budget
    if total_annotated >= config["annotation_budget"]:
        print("Stopping: annotation budget exhausted.")
        return False

    # Check max rounds
    if current_round >= config["num_rounds"]:
        print("Stopping: maximum rounds reached.")
        return False

    # Check for plateau (diminishing returns)
    if len(learning_curve) >= 3:
        recent = learning_curve[-3:]
        improvements = [
            recent[i+1]["val_f1"] - recent[i]["val_f1"]
            for i in range(len(recent) - 1)
        ]
        if all(imp < 0.005 for imp in improvements):
            print("Stopping: performance plateau detected.")
            return False

    return True
```

## Stopping Criteria

The loop stops when any of these conditions is met:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Budget exhausted | `total_annotated >= annotation_budget` | No more annotation resources |
| Max rounds reached | `current_round >= num_rounds` | Predetermined limit |
| Performance plateau | F1 improvement < 0.005 for 3 consecutive rounds | Diminishing returns |
| Target metric reached | F1 >= target_f1 (if set) | Model is good enough |
| Annotator request | Expert signals to stop | Domain expert judgment |

## Complete Loop Assembly

```python
def run_active_learning_loop(config):
    """Run the complete active learning loop."""
    # Step 1: Initialize
    model, unlabeled_pool, train_annotations, device = initialize(config)
    learning_curve = []
    total_annotated = len(train_annotations)

    for round_num in range(1, config["num_rounds"] + 1):
        print(f"\n{'='*60}")
        print(f"ACTIVE LEARNING ROUND {round_num}")
        print(f"{'='*60}")

        # Step 2: Predict (skip if no model)
        predictions = {}
        if model is not None:
            predictions = predict_on_pool(model, unlabeled_pool, device)

        # Step 3: Select
        selected = select_samples(config, unlabeled_pool, predictions)

        # Step 4: Export to annotation tool
        export_to_annotation_tool(selected, predictions, config)

        # Step 5: Wait for human annotation
        input(f"Press Enter when annotation for round {round_num} is complete...")

        # Step 6: Download corrected annotations
        new_annotations = download_annotations(config, round_num)

        # Step 7: Merge
        train_annotations = merge_annotations(train_annotations, new_annotations,
                                               config["output_dir"] / "train.json")
        total_annotated += len(new_annotations)

        # Remove annotated images from pool
        unlabeled_pool = [p for p in unlabeled_pool if str(p) not in new_annotations]

        # Step 8: Retrain
        model = retrain_model(config, train_annotations, round_num)

        # Step 9: Evaluate
        evaluate_round(model, val_dataset, round_num, total_annotated, learning_curve)

        # Step 10: Continue?
        if not should_continue(config, learning_curve, round_num, total_annotated):
            break

    # Save learning curve
    save_learning_curve(learning_curve, config["output_dir"] / "learning_curve.csv")
    print(f"\nActive learning complete after {len(learning_curve)} rounds.")
    print(f"Total annotations: {total_annotated}")
    print(f"Final F1: {learning_curve[-1]['val_f1']:.3f}")

    return model, learning_curve
```

## Learning Curve Logging

After each round, log the following to a CSV file:

```csv
round,n_annotations,val_f1,val_precision,val_recall,val_mae,strategy,timestamp
1,200,0.421,0.512,0.358,12.3,rgb_contrast,2025-01-15T10:30:00
2,400,0.587,0.623,0.554,8.1,logit_uncertainty,2025-01-16T14:20:00
3,600,0.651,0.678,0.626,6.4,logit_uncertainty,2025-01-17T09:45:00
```

This data enables plotting the learning curve to visualize the relationship between
annotation effort and model performance, which is the central diagnostic tool for
evaluating whether active learning is working effectively.
