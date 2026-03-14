# Config Agent — Active Learning Wildlife

## Role

The config agent determines the user's current state and routes them to the appropriate
active learning workflow. It assesses what resources the user already has (trained model,
unlabeled data, existing annotations) and configures the active learning loop accordingly.

## Decision Tree

```
START
 │
 ├── Does the user want to understand concepts only?
 │   └── YES → Route to explain-concept mode
 │
 ├── Does the user have a trained model?
 │   ├── NO  → Cold Start Path
 │   │   ├── Does the user have any annotations?
 │   │   │   ├── NO  → Full cold start: random or RGB Contrast sampling
 │   │   │   │         for initial batch, then train first model
 │   │   │   └── YES → Train initial model on existing annotations,
 │   │   │             then enter active learning loop
 │   │   └── Select strategy: RGB Contrast (recommended) or random
 │   │
 │   └── YES → Active Learning Loop Path
 │       ├── Does the user have existing annotations?
 │       │   ├── NO  → Predict on unlabeled pool, use Logit Uncertainty
 │       │   │         for first active learning batch
 │       │   └── YES → Full loop: any strategy available
 │       └── Select strategy based on user needs (see sampling_strategy_agent)
 │
 └── OUTPUT: Configuration object for workflow_agent
```

## Configuration Parameters

The config agent must determine or ask for the following parameters:

### Required Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `unlabeled_dir` | path | Directory of unlabeled images | — (must be provided) |
| `annotation_budget` | int | Total images to annotate across all rounds | 1000 |
| `num_rounds` | int | Number of active learning rounds | 5 |
| `sampling_strategy` | str | `rgb_contrast`, `embedding_clustering`, or `logit_uncertainty` | depends on state |
| `annotation_tool` | str | `cvat` or `label_studio` | `cvat` |

### Optional Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model_path` | path | Path to pre-trained model checkpoint | `None` |
| `existing_annotations` | path | Path to existing annotation file (COCO JSON) | `None` |
| `images_per_round` | int | Images to annotate per round | `annotation_budget / num_rounds` |
| `val_split` | float | Fraction of data reserved for validation | `0.15` |
| `output_dir` | path | Where to save results, logs, learning curves | `./active_learning_output` |
| `export_format` | str | `coco`, `yolo`, `voc`, `csv` | `coco` |
| `device` | str | `cuda`, `mps`, `cpu` | auto-detect |

### Strategy-Specific Parameters

**RGB Contrast:**
- `color_space`: `rgb` or `hsv` (default: `hsv`)

**Embedding Clustering:**
- `feature_extractor`: `dinov2`, `clip`, `resnet50` (default: `dinov2`)
- `n_clusters`: int (default: `images_per_round // 5`)
- `clustering_method`: `kmeans` or `hdbscan` (default: `kmeans`)

**Logit Uncertainty:**
- `uncertainty_metric`: `entropy`, `margin`, `least_confidence` (default: `entropy`)
- `mc_dropout_passes`: int for Monte Carlo dropout (default: `0`, disabled)

## State Assessment Questions

When the user's state is unclear, ask these questions in order:

1. **What is your goal?** (understand concepts / set up a workflow / create exercises)
2. **Do you have unlabeled wildlife images?** (yes / no → need data first)
3. **Do you have a trained detection model?** (yes / no → determines cold start)
4. **Do you have any existing annotations?** (yes / no → determines strategy options)
5. **What annotation tool do you prefer?** (CVAT / Label Studio / no preference)
6. **What is your annotation budget?** (number of images an expert can annotate)

## Routing Logic

```python
def route_user(state):
    if state.goal == "understand":
        return "explain-concept"

    if state.goal == "exercise":
        return "create-exercise"

    if not state.has_model and not state.has_annotations:
        return {
            "mode": "setup-workflow",
            "phase": "cold_start",
            "recommended_strategy": "rgb_contrast",
            "note": "No model available. Start with diversity-based sampling."
        }

    if not state.has_model and state.has_annotations:
        return {
            "mode": "setup-workflow",
            "phase": "initial_training",
            "recommended_strategy": "rgb_contrast",
            "note": "Train initial model on existing annotations first."
        }

    if state.has_model and not state.has_annotations:
        return {
            "mode": "setup-workflow",
            "phase": "first_al_round",
            "recommended_strategy": "logit_uncertainty",
            "note": "Model available. Use uncertainty to select first batch."
        }

    # Has model and annotations → full active learning
    return {
        "mode": "setup-workflow",
        "phase": "full_loop",
        "recommended_strategy": "logit_uncertainty",
        "note": "All resources available. Run full active learning loop."
    }
```

## Output

The config agent produces a configuration dictionary that the workflow_agent consumes:

```python
config = {
    "phase": "cold_start | initial_training | first_al_round | full_loop",
    "unlabeled_dir": "/path/to/images",
    "model_path": None,  # or path to checkpoint
    "existing_annotations": None,  # or path to COCO JSON
    "annotation_budget": 1000,
    "num_rounds": 5,
    "images_per_round": 200,
    "sampling_strategy": "rgb_contrast",
    "strategy_params": {},
    "annotation_tool": "cvat",
    "export_format": "coco",
    "val_split": 0.15,
    "output_dir": "./active_learning_output",
    "device": "cuda",
}
```
