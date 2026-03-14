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


## GEneral Knowledge
# The complete guide to MegaDetector in 2025

**MegaDetector exists in two actively maintained but divergent ecosystems**, each led by different teams with different philosophies. The agentmorris/MegaDetector repository — maintained by original developer Dan Morris — remains the go-to for batch processing camera trap and drone imagery at scale, while Microsoft's repurposed CameraTraps repo (now PytorchWildlife) offers a broader AI-for-conservation platform with its own model family. Understanding which to use, and how, depends entirely on your workflow needs. This report maps the full landscape: repositories, model versions, inference pipelines, training approaches, ecosystem tools, and specific guidance for drone imagery.

---

## Two repos, two philosophies, one shared origin

MegaDetector was born at Microsoft AI for Earth, primarily developed by Dan Morris. When Morris left Microsoft in 2023, the project split into two independent tracks that continue to evolve in parallel.

**agentmorris/MegaDetector** is the direct continuation of the original MegaDetector project. It was forked from microsoft/CameraTraps on May 20, 2023 and is actively maintained by Morris (now at Google AI for Nature). As of mid-2025, it has **~250 stars**, **5,015 commits**, and only **1 open issue** — a sign of tight maintenance. The latest release is **MDv1000** (July 23, 2025), a family of five new detection models. It is installable via `pip install megadetector` and documented at megadetector.readthedocs.io. This repo focuses exclusively on MegaDetector detection tools: batch inference scripts, postprocessing utilities, format conversion, and Repeat Detection Elimination. For species classification, it recommends Google's standalone SpeciesNet.

**microsoft/CameraTraps (PytorchWildlife)** was repurposed by the Microsoft AI for Good Lab team (Zhongqi Miao, Andres Hernandez, and others) into a broader collaborative framework. It retains **~990 stars** and the original URL, but its identity has fundamentally changed. The latest release is **PytorchWildlife v1.2.4.2** (October 2025), installable via `pip install PytorchWildlife`. It provides a model zoo encompassing MegaDetector V5 and V6, DeepFaune, HerdNet (for aerial imagery), and multiple species classifiers. Old MegaDetector code is preserved in an archive branch.

Neither repo claims the other is deprecated. Both acknowledge each other explicitly. For **new users in 2025**, the practical recommendation depends on use case:

- **For batch processing at scale** (thousands to millions of images, production pipelines, drone imagery with tiling): use **agentmorris/MegaDetector**
- **For a broader AI conservation platform** with multiple model types, built-in classification, and a Gradio demo: use **PytorchWildlife**
- **For a no-code GUI** (the way **95% of users** run MegaDetector): use **AddaxAI** (formerly EcoAssist), which both repos recommend

Other notable repositories include **AddaxAI** itself (github.com/PetervanLunteren/AddaxAI), Google's **SpeciesNet** (github.com/google/cameratrapai), and several domain-specific fine-tuned models like the USGS Tegu detector and UNSW Goanna detector.

---

## Model versions span three architectural generations

MegaDetector's evolution traces a clear arc from TensorFlow Faster-RCNN through YOLOv5 to a diverse family of modern architectures. **MDv5a remains the default model** in the agentmorris repo, though MDv1000 offers incremental improvements.

| Version | Architecture | Source repo | Training data | Typical threshold | License concern |
|---------|-------------|-------------|---------------|-------------------|-----------------|
| MDv2–v4.1 | Faster-RCNN (TF) | Legacy | Camera traps | 0.7–0.8 | None |
| **MDv5a** (recommended) | **YOLOv5x6** | agentmorris | Camera traps + COCO | **0.15–0.25** | GPL (YOLOv5) |
| MDv5b | YOLOv5x6 | agentmorris | Camera traps only | 0.15–0.25 | GPL (YOLOv5) |
| MDv6 (compact/extra) | YOLOv9, YOLOv10, RT-DETR | microsoft/CameraTraps | Camera traps | Varies | AGPL or MIT/Apache |
| **MDv1000-redwood** | **YOLOv5x6** | agentmorris | Pre-trained + camera traps | 0.15–0.25 | GPL (YOLOv5) |
| MDv1000-cedar | YOLOv9c | agentmorris | Same | Similar | GPL |
| MDv1000-larch | YOLOv11L | agentmorris | Same | Similar | AGPL (Ultralytics) |
| MDv1000-sorrel | YOLOv11s | agentmorris | Same | Similar | AGPL (Ultralytics) |
| MDv1000-spruce | YOLOv5s | agentmorris | Same | Similar | GPL |

All models detect exactly **three categories**: animal (1), person (2), vehicle (3). Species-level classification requires a separate model like SpeciesNet or DeepFaune.

**MDv5a** was trained on several million camera trap images plus the COCO dataset, giving it better generalization to reptiles, birds, and unusual objects. **MDv5b** was trained only on camera trap data and may perform marginally better on very dark or low-contrast images. The **MDv1000 family** (July 2025) adds five models named after trees, spanning different speed/accuracy tradeoffs. MDv1000-redwood matches MDv5's architecture (YOLOv5x6) with updated training; MDv1000-cedar runs **~2× faster** using YOLOv9c. Morris deliberately avoided calling these "MDv6" to prevent confusion with Microsoft's independent MDv6 release.

All model weights auto-download on first use. They are hosted on GitHub Releases, HuggingFace (agentmorris/megadetector), and Kaggle. The critical threshold difference: **MDv5/v1000 use thresholds of 0.15–0.25**, vastly different from MDv4's 0.7–0.8. Getting this wrong will produce either floods of false positives or missed detections.

---

## Inference at every scale, from single images to millions

The agentmorris/MegaDetector repo provides a mature, well-documented inference stack with three tiers: Python API, CLI, and large-batch orchestration.

**For programmatic use**, the Python API is clean and straightforward:

```python
from megadetector.detection.run_detector_batch import load_and_run_detector_batch, write_results_to_file
from megadetector.utils import path_utils

results = load_and_run_detector_batch('MDV5A', path_utils.find_images('/data/images', recursive=True))
write_results_to_file(results, 'output.json', relative_path_base='/data/images', detector_file='MDV5A')
```

For single-image inference, `run_detector.load_detector('MDV5A')` returns a model object with a `generate_detections_one_image()` method. Models auto-download on first invocation.

**For command-line batch processing**, `run_detector_batch.py` is the workhorse:

```bash
python -m megadetector.detection.run_detector_batch MDV5A /path/to/images output.json \
  --recursive --checkpoint_frequency 10000 --resume_from_checkpoint output_checkpoint.json
```

Key flags include `--augment` for test-time augmentation (substantially helps with difficult images), `--output_relative_filenames`, and `--include_image_timestamp`. The script supports single-GPU inference; multi-GPU is achieved by splitting image lists across parallel processes.

**For production-scale runs** (20TB of drone imagery, for instance), `manage_local_batch.py` orchestrates the full pipeline: splitting images into chunks for multi-GPU processing, running inference, performing Repeat Detection Elimination (RDE) to remove persistent false positives from static objects, generating HTML preview pages via `postprocess_batch_results.py`, and splitting/merging output JSON files. This is the tool for datasets of **1 million+ images**.

### GPU and CPU performance benchmarks

| Hardware | Images/sec | Images/day |
|----------|-----------|------------|
| **RTX 4090** | **~17.6** | **~1,500,000** |
| RTX 3090 | ~11.4 | ~985,000 |
| RTX 3080 | ~9.5 | ~820,000 |
| M3 MacBook Pro (18 GPU cores) | ~4.6 | ~398,000 |
| CPU only (i7-13700K) | ~0.8 | ~69,000 |

MDv1000-cedar is approximately **2× faster** than MDv5/MDv1000-redwood at similar accuracy. On CPU alone, a decent 2025 laptop processes **30,000–50,000 images/day** — feasible for small projects but impractical for large drone surveys.

### Output format: the MD results JSON

All MegaDetector tools produce a standardized JSON format (version 1.5, spec at lila.science/megadetector-output-format). Bounding boxes use **normalized coordinates** `[x_min, y_min, width, height]` with values 0–1 relative to image dimensions — notably, this is *not* standard COCO format. The `detector_metadata` field embeds recommended confidence thresholds that downstream tools like Timelapse consume automatically. A format validator ships with the repo.

**Postprocessing tools** include `separate_detections_into_folders.py` (sorts images into animal/person/vehicle/empty folders), `subset_json_detector_output.py` (filters and splits JSON files), `compare_batch_results.py`, and `merge_detections.py` (combines results from multiple model runs).

---

## Tiling transforms drone imagery performance

MegaDetector resizes all images to **1280 pixels** on the longest axis before inference. For high-resolution drone imagery where animals may occupy only a few dozen pixels, this downscaling makes small subjects invisible. The **tiling pipeline** (`megadetector.detection.run_tiled_inference`) solves this by splitting images into overlapping 1280×1280 tiles, running detection on each independently, then merging and de-duplicating results.

A 2024 study validated this approach in Arctic conditions: tiling with MDv5a detected reindeer as small as **18 pixels** — well below the previously reported 60-pixel minimum. The study found tiling particularly effective for animals at distance and those camouflaged against snow. However, tiling has important caveats. Very large animals spanning multiple tiles may be detected multiple times or missed entirely. The recommended approach is to run **both tiled and whole-image inference** and merge results using `merge_detections.py`. Current tiling writes tiles to disk, requiring temporary storage roughly equal to input data size; in-memory tiling is on the development roadmap.

Test-time augmentation (`--augment`) complements tiling by creating multiple transformed copies of each image before detection, helping with low-light and low-contrast conditions. For drone surveys, **combining tiling + TTA + merging results from MDv5a and MDv5b** produces the most comprehensive detections.

Note that tiling is available only via the agentmorris/MegaDetector CLI — AddaxAI does not yet support it natively (open issue #83).

---

## Fine-tuning follows standard YOLO workflows

MegaDetector can be fine-tuned on custom datasets, and real-world examples demonstrate this works well for domain-specific applications. Two parallel approaches exist, corresponding to the two repos.

**For MDv5/MDv1000 (agentmorris repo)**: Fine-tuning uses standard YOLOv5 training with MegaDetector weights as the starting checkpoint. The official Kaggle notebook (kaggle.com/code/agentmorris/fine-tuning-megadetector) walks through the process. The data pipeline follows a well-established pattern:

1. Run MegaDetector on your dataset to bootstrap annotations
2. Convert MD results to labelme format for manual review and correction
3. Convert corrected annotations to COCO Camera Traps JSON format
4. Use `megadetector/data_management/coco_to_yolo.py` to produce YOLO-format labels
5. Train with YOLOv5:

```bash
python train.py --weights md_v5a.0.0.pt --data dataset.yaml --img 1280 --batch 8 --epochs 300
```

The YOLO training format requires paired `images/` and `labels/` directories with train/val splits, where each label `.txt` file contains `class_id x_center y_center width height` (all normalized 0–1), plus a `dataset.yaml` specifying paths and class names.

**For MDv6 (PytorchWildlife)**: The `PW_FT_detection` module (released January 2025 in v1.2.0) provides official fine-tuning from any MDv6 pretrained checkpoint using the Ultralytics framework. Fine-tuned models integrate directly back into the PytorchWildlife pipeline.

**Documented success stories** include the USGS Tegu detector (Florida invasive species, YOLOv5x6 from MDv5a), the UNSW Goanna detector (5 Australian species, both YOLOv5 and YOLOv8 variants), TRAPPER AI (18 European mammals, YOLOv8-m), and DeepFaune (European wildlife, integrated into PytorchWildlife v1.2.1). The repos at github.com/agentmorris/usgs-tegus and github.com/agentmorris/unsw-goannas provide complete, reproducible end-to-end training pipelines with every script included.

For converting *back* from MD output to YOLO training format, `megadetector/data_management/yolo_output_to_md_output.py` handles the YOLOv5 detect.py output direction, while `coco_to_yolo.py` and `yolo_to_coco.py` handle bidirectional COCO↔YOLO conversion.

---

## The broader ecosystem connects detection to decisions

**AddaxAI** (formerly EcoAssist) is the dominant GUI entry point, described by the MegaDetector docs as how "95% of users" run the model. It provides no-code access to MDv5a/v5b detection, species classification via DeepFaune/SpeciesNet/regional models, built-in annotation editing, confidence threshold filtering, and even no-code custom YOLOv5 training. Published in JOSS (2023), it runs on Windows, Mac, and Linux.

**Timelapse2** is the primary tool for *reviewing* MegaDetector results. It imports the MD JSON file, displays bounding boxes overlaid on images, and lets researchers filter by classification, sort by confidence, and populate data fields with detection counts. It supports both the agentmorris (v5) and Microsoft (v6) output formats.

**WildTrax** (Alberta Biodiversity Monitoring Institute) integrates MDv5 as its core auto-tagger across **10+ million images**, adding custom models for cattle detection, staff photo identification, and automatic human-image blurring.

**SpeciesNet** (Google) is the recommended species classifier in the agentmorris ecosystem. It classifies **~2,000 species** globally and produces output convertible to MD format via included scripts. For PytorchWildlife users, the model zoo includes its own classifiers (AI4G Amazon Rainforest, Serengeti, etc.).

**HerdNet**, integrated into PytorchWildlife v1.1.0, deserves special attention for aerial work — it is a **point-based overhead animal detection model** specifically designed for drone imagery, trained on the Ennedi 2019 dataset. The PytorchWildlife team is also developing a bounding-box-based aerial detection model.

Both repos have **PyPI packages** (`megadetector` and `PytorchWildlife`), and a lightweight `megadetector-utils` package exists for users who need only postprocessing tools without PyTorch dependencies. HuggingFace hosts MDv5 weights at agentmorris/megadetector and interactive demos at multiple Spaces.

---

## Practical workflow recommendations for drone researchers

**For running inference on a large drone image dataset**, the single best workflow in 2025 is:

1. Install via `pip install megadetector`
2. Run whole-image inference: `python -m megadetector.detection.run_detector_batch MDV5A /data/drone_images output_whole.json --recursive --checkpoint_frequency 10000`
3. Run tiled inference using `run_tiled_inference.py` to catch small/distant animals
4. Merge results with `merge_detections.py` to combine whole-image and tiled detections
5. Run Repeat Detection Elimination if cameras have fixed positions
6. Review in Timelapse2 or filter programmatically using confidence thresholds (start at **0.15** and adjust)

For **20TB-scale datasets**, use `manage_local_batch.py` to orchestrate multi-GPU processing, splitting the image set across available GPUs. An RTX 4090 processes ~1.5 million images/day; budget your timeline accordingly.

**For fine-tuning on aerial wildlife data**, the recommended approach is:

1. Run MegaDetector on a representative subset to bootstrap bounding box annotations
2. Manually review and correct annotations in labelme (the agentmorris/labelme fork integrates well)
3. Add annotations for missed detections and domain-specific classes
4. Convert to YOLO format using `coco_to_yolo.py`
5. Fine-tune from MDv5a weights using YOLOv5 training at image size 1280
6. Alternatively, consider **HerdNet** (via PytorchWildlife) if your task involves counting animals from directly overhead — it was purpose-built for this perspective

**Known limitations for drone/aerial imagery** are significant. MegaDetector was trained on ground-level camera trap images; overhead perspectives are underrepresented. Human reviewers detected objects **15× smaller** than MegaDetector in one Arctic study. Performance on time-lapse drone captures (where animals are often distant) dropped to **≤61.6% accuracy** versus **≥94.6%** on motion-triggered ground-level images. Tiling mitigates but does not fully solve the resolution problem. For serious aerial wildlife surveys, fine-tuning on domain-specific data or using HerdNet alongside MegaDetector is strongly recommended.

## Conclusion

The MegaDetector ecosystem in 2025 is healthier than ever but genuinely split. The agentmorris/MegaDetector repo offers the most complete, production-ready toolchain for batch detection with unmatched postprocessing capabilities and the new MDv1000 model family. PytorchWildlife provides a more expansive platform with MDv6, integrated classification, and aerial-specific models like HerdNet. For drone imagery specifically, the combination of tiled inference, test-time augmentation, and merged multi-pass results from the agentmorris repo represents the current state of the art — but fine-tuning on domain-specific aerial data yields the largest accuracy gains. The critical insight is that MegaDetector is not one tool but an ecosystem, and the best results come from combining multiple models, inference strategies, and postprocessing pipelines rather than relying on any single configuration.