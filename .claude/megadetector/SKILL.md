---
name: megadetector
description: "Slim coding skill for MegaDetector wildlife detection. Generates working code examples for single-image, batch, and folder-based animal detection using the official MegaDetector repository. Covers confidence threshold selection, output format (normalized bounding boxes), batch processing with checkpointing, and the MegaDetector-then-classifier pipeline pattern. Intentionally minimal — MegaDetector is a wrapper, not a training framework."
metadata:
  version: "1.0"
  last_updated: "2026-03-13"
  source_repo: "/Users/christian/work/hnee/MegaDetector"
---

# MegaDetector — Slim Wildlife Detection Skill

## Quick Start

```
User: "Show me how to run MegaDetector on a folder of camera trap images"
→ megadetector_code_agent generates a working script using load_detector + run_detector_batch

User: "I want to detect animals then classify species"
→ pipeline_integration_agent generates MD → crop → timm classifier pipeline
```

## Trigger Keywords

MegaDetector, PytorchWildlife, MDV5A, MDV5B, animal detection, camera trap detection,
empty image filter, detect animals, wildlife detection pre-filter, animal present absent,
run_detector, run_detector_batch, camera trap AI, wildlife detection wrapper

## Does NOT Trigger

| If the user wants... | Use this skill instead |
|---|---|
| Species classification (not just animal/person/vehicle) | wildlife-classification |
| Detection on large orthomosaics / drone imagery | sahi-inference |
| Point-based density detection (HerdNet) | herdnet-training |
| Fine-tuning a detection model | herdnet-training or wildlife-classification |

## Agent Team

| Agent | Role | Phase |
|---|---|---|
| `megadetector_code_agent` | Produces working code for single-image, batch, and folder-based detection using the official MegaDetector API | Primary |
| `pipeline_integration_agent` | Designs and codes the MD → species classifier pipeline: crop animals from bounding boxes, feed to classifier, merge results | Integration |
| `exercise_designer_agent` | Produces minimal student exercises (activated only in `create-exercise` mode) | Optional |

## Orchestration

```
User Input
    |
=== Phase 0: INTAKE ===
    |
    Determine mode: quickstart / pipeline / explain / exercise
    |
=== Phase 1: CODE GENERATION ===
    |
    |-> [megadetector_code_agent]
    |   - Load detector: load_detector('MDV5A')
    |   - Single image or batch inference
    |   - Results extraction to DataFrame
    |   - Confidence filtering
    |   Output: Working Python script
    |
    +-> (if pipeline mode)
        [pipeline_integration_agent]
        - Crop bounding boxes from detections
        - Feed crops to timm/DeepFaune/SpeciesNet classifier
        - Merge detection + classification results
        Output: End-to-end detect-then-classify script
    |
=== Phase 2: EXERCISE (optional) ===
    |
    |-> [exercise_designer_agent]
        Output: Scaffolded notebook exercise
```

## Operational Modes

| Mode | Description | Agents Active |
|---|---|---|
| `quickstart` (default) | Minimal working code for described use case | megadetector_code_agent |
| `explain-concept` | Explain how MegaDetector works, what it can/cannot do | megadetector_code_agent |
| `pipeline` | Full detect-then-classify pipeline | megadetector_code_agent + pipeline_integration_agent |
| `create-exercise` | Student exercise | exercise_designer_agent |

## Key Facts (agents must know these)

### Model Versions
- **MDV5A** (recommended): YOLOv5, image_size=1280, confidence threshold 0.2
- **MDV5B**: Alternative training data, same architecture
- **v1000 series** (2025): Latest, 5 named variants (redwood/cedar/larch/sorrel/spruce)
- Default string `"MDV5A"` auto-downloads model to cache

### Output Format
- JSON with `format_version: "1.5"`
- Bounding boxes: **normalized** `[x_min, y_min, width, height]` (0–1 range, NOT pixels)
- Classes: `{'1': 'animal', '2': 'person', '3': 'vehicle'}`
- Confidence: 0.0–1.0; typical threshold **0.2** for wildlife

### The 3-Line Pattern
```python
from megadetector.detection.run_detector import load_detector
detector = load_detector('MDV5A')
results = detector.generate_detections_one_batch(transform(img), img.shape, img_path)
```

### Batch Processing
```bash
python -m megadetector.detection.run_detector_batch MDV5A "folder/" "output.json" \
  --output_relative_filenames --recursive --checkpoint_frequency 10000
```

## References

| File | Content |
|---|---|
| `references/megadetector_api.md` | Core API: load_detector, generate_detections_one_batch, run_detector_batch CLI |
| `references/output_format_reference.md` | Full JSON schema, normalized bbox convention, failure field |
| `references/confidence_threshold_guide.md` | Threshold selection: 0.2 typical, 0.05 conservative, sweep methodology |
| `references/md_to_classifier_pipeline.md` | Detect → crop → classify architecture pattern |
| `references/shared_pytorch_conventions.md` | Device selection, torch.load, Path objects |

## Templates

| File | Purpose |
|---|---|
| `templates/megadetector_quickstart_template.py` | Minimal working script (≤30 lines) |
| `templates/md_classifier_pipeline_template.py` | Full detect-then-classify pipeline |

## Failure Paths

| ID | Trigger | Handling |
|---|---|---|
| F1 | Model download fails (network) | Instruct user to manually download .pt file and pass path |
| F2 | CUDA OOM on batch | Reduce batch size; suggest CPU fallback for small jobs |
| F3 | Student tries to fine-tune MD | Redirect to herdnet-training or wildlife-classification skill |
| F4 | Normalized bbox confusion | Explain [x,y,w,h] normalized format; show pixel conversion |
