# HILDA Framework Reference (v0.3.0)

## Overview

HILDA is an active learning framework for wildlife detection, developed at HNEE.
It provides sampling strategies, annotation tool connectors, and a loop orchestrator
for iterative model improvement with minimal human annotation effort.

**Source:** `/Users/christian/PycharmProjects/hnee/active_learning`

---

## Core Components

### Sampling Strategies

HILDA implements three sampling strategies, each targeting a different phase
of the active learning loop:

| Strategy | Class | Module | Requirements |
|----------|-------|--------|--------------|
| RGB Contrast | `RGBContrastSampler` | `hilda.sampling.rgb_contrast` | No model, no GPU |
| Embedding Clustering | `EmbeddingClusterSampler` | `hilda.sampling.embedding_cluster` | Feature extractor, GPU |
| Logit Uncertainty | `UncertaintySampler` | `hilda.sampling.uncertainty` | Trained model, GPU |

### Strategy Details

**RGB Contrast:**
- Computes color histograms in HSV space (configurable bins)
- Uses farthest-point sampling for maximum visual diversity
- Best for cold start (no model available)
- Parameters: `n_samples`, `color_space` ('rgb' or 'hsv'), `bins`

**Embedding Clustering:**
- Extracts embeddings from a pre-trained vision model (DINOv2, CLIP, ResNet50)
- Clusters using K-Means or HDBSCAN
- Selects representative images from each cluster + farthest-point for diversity
- Parameters: `n_samples`, `feature_extractor`, `n_clusters`, `clustering_method`

**Logit Uncertainty:**
- Runs model inference on unlabeled pool
- Scores images by uncertainty metric (entropy, margin, least_confidence)
- Selects top-K most uncertain images
- Optional: MC Dropout for better uncertainty estimates
- Parameters: `n_samples`, `model`, `uncertainty_metric`, `mc_dropout_passes`

---

## Annotation Tool Connectors

HILDA provides connectors for CVAT and Label Studio:

### CVAT Connector
- Uses `cvat-sdk` for automation
- Creates tasks, uploads images, imports pre-annotations
- Downloads corrected annotations in COCO format
- Supports bounding box and point annotations

### Label Studio Connector
- Uses `label-studio-sdk` for automation
- Creates projects with detection labeling config
- Uploads images with prediction pre-annotations
- Downloads annotations in COCO, YOLO, or VOC format

---

## Loop Orchestrator

The loop orchestrator manages the active learning cycle:

```python
from hilda.loop import ActiveLearningLoop

loop = ActiveLearningLoop(
    model=model,
    unlabeled_pool=image_paths,
    annotation_tool='cvat',
    sampling_strategy='logit_uncertainty',
    images_per_round=200,
    max_rounds=5,
    annotation_budget=1000,
    output_dir='./al_output',
)

loop.run()
```

The orchestrator:
1. Calls the configured sampling strategy to select images
2. Exports selected images + pre-annotations to the annotation tool
3. Waits for expert annotation completion
4. Downloads corrected annotations
5. Merges with existing training set
6. Retrains the model
7. Evaluates on held-out validation set
8. Logs learning curve data
9. Checks stopping criteria (budget, plateau, max rounds)
10. Repeats or stops

---

## Key Findings from HILDA Research

### HITL vs. Expert-Only Annotation
- Human-in-the-loop (HITL) with model pre-annotations catches **22-30% more animals**
  than expert-only annotation in dense iguana colonies
- Experts systematically undercount in dense aggregations
- Model predictions serve as a "counting aid" that draws attention to missed individuals

### Annotation Efficiency
- Pre-annotations reduce annotation time by 3-5x compared to blank-image annotation
- The largest gains come from the first 2-3 active learning rounds
- After round 3-5, diminishing returns typically set in

### Strategy Effectiveness (Empirical)
- RGB Contrast is surprisingly effective for cold start (comparable to random
  at round 1, but better coverage of visual diversity)
- Logit Uncertainty consistently outperforms random sampling from round 2 onward
- Embedding Clustering is best when the unlabeled pool contains many distinct
  visual domains (different habitats, lighting conditions, camera angles)

---

## Installation and Dependencies

```bash
# Core
pip install torch torchvision
pip install scikit-learn
pip install Pillow
pip install numpy

# For Embedding Clustering
pip install timm  # or torch.hub for DINOv2

# For CVAT integration
pip install cvat-sdk

# For Label Studio integration
pip install label-studio-sdk

# For visualization
pip install matplotlib
pip install seaborn
```

---

## File Structure

```
hilda/
├── __init__.py
├── loop.py                    # ActiveLearningLoop orchestrator
├── sampling/
│   ├── __init__.py
│   ├── base.py                # BaseSampler abstract class
│   ├── rgb_contrast.py        # RGBContrastSampler
│   ├── embedding_cluster.py   # EmbeddingClusterSampler
│   └── uncertainty.py         # UncertaintySampler
├── connectors/
│   ├── __init__.py
│   ├── cvat.py                # CVAT SDK wrapper
│   └── label_studio.py        # Label Studio SDK wrapper
├── utils/
│   ├── __init__.py
│   ├── learning_curve.py      # Logging and plotting
│   └── format_conversion.py   # COCO ↔ YOLO ↔ CSV converters
└── config.py                  # Configuration dataclass
```
