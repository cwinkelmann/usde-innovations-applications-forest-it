# Academic Research & Wildlife AI Skills

A suite of Claude Code skills for academic research, paper writing, peer review, pipeline orchestration, and aerial wildlife detection course development.

## Skills Overview

### Academic Research Skills

| Skill | Purpose | Key Modes |
|-------|---------|-----------|
| `deep-research` v2.2 | Universal 10-agent research team | full, quick, socratic, review, lit-review, fact-check |
| `academic-paper` v2.2 | 10-agent academic paper writing | full, plan, outline-only, revision, abstract-only, lit-review, format-convert, citation-check |
| `academic-paper-reviewer` v1.3 | Multi-perspective paper review (5 reviewers) | full, re-review, quick, methodology-focus, guided |
| `academic-pipeline` v2.2 | Full pipeline orchestrator | (coordinates all above) |

### Wildlife AI Course Skills

| Skill | Purpose | Key Modes |
|-------|---------|-----------|
| `megadetector` v1.0 | MegaDetector detection code examples | generate-code, batch-processing, pipeline, create-exercise |
| `wildlife-classification` v1.0 | timm/DeepFaune/SpeciesNet fine-tuning with catastrophic forgetting mitigation | generate-code, explain-concept, evaluate-model, create-exercise, compare-models, full-course-module |
| `practical-cv-wildlife` v1.0 | PCV-to-wildlife bridge (fills curriculum gaps) | map-curriculum, generate-aerial-module, fill-detection-gap, adapt-notebook, create-exercise, full-course-module |
| `herdnet-training` v1.0 | HerdNet/animaloc point detection training lifecycle | train-from-scratch, fine-tune, inference-only, diagnose-training, explain-concept, create-exercise, full-course-module |
| `sahi-inference` v1.0 | SAHI tiled inference for large drone/satellite imagery | generate-code, explain-concept, optimize-pipeline, create-exercise, full-course-module |
| `active-learning-wildlife` v1.0 | Active learning loop with HILDA (CVAT/Label Studio) | setup-workflow, explain-concept, select-strategy, create-exercise, full-course-module |

## Routing Rules

### Academic Skills Routing

1. **academic-pipeline vs individual skills**: academic-pipeline = full pipeline orchestrator (research → write → review → revise → finalize). If the user only needs a single function (just research, just write, just review), trigger the corresponding skill directly without the pipeline.

2. **deep-research vs academic-paper**: Complementary. deep-research = upstream research engine (investigation + fact-checking), academic-paper = downstream publication engine (paper writing + bilingual abstracts). Recommended flow: deep-research → academic-paper.

3. **deep-research socratic vs full**: socratic = guided Socratic dialogue to help users clarify their research question. full = direct production of research report. When the user's research question is unclear, suggest socratic mode.

4. **academic-paper plan vs full**: plan = chapter-by-chapter guided planning via Socratic dialogue. full = direct paper production. When the user wants to think through their paper structure, suggest plan mode.

5. **academic-paper-reviewer guided vs full**: guided = Socratic review that engages the author in dialogue about issues. full = standard multi-perspective review report. When the user wants to learn from the review, suggest guided mode.

### Wildlife AI Skills Routing

6. **megadetector**: Trigger for MegaDetector detection code, batch processing, confidence thresholds, or MD→classifier pipeline. Does NOT cover species classification training — use `wildlife-classification` for that.

7. **wildlife-classification**: Trigger for species classification fine-tuning (timm, DeepFaune, SpeciesNet), catastrophic forgetting, discriminative learning rates, or model comparison. Does NOT cover object detection — use `megadetector` or `herdnet-training`.

8. **practical-cv-wildlife**: Trigger for bridging PCV course content to wildlife domain, adapting existing notebooks, creating aerial imagery modules, or filling the detection implementation gap (YOLOv8). Does NOT re-teach PCV content.

9. **herdnet-training**: Trigger for HerdNet/animaloc point-based detection, FIDT maps, Hydra configs, stitcher/LMDS inference, or training diagnosis. Most complex skill — covers the full training lifecycle.

10. **sahi-inference**: Trigger for SAHI tiled inference on large images, slice parameter tuning, overlap optimization, NMS/NMM postprocessing, or GeoTIFF coordinate conversion. Wraps any SAHI-supported detector (YOLOv8, MegaDetector, Detectron2). Does NOT train models — use `herdnet-training` or `wildlife-classification` for training.

11. **active-learning-wildlife**: Trigger for active learning loops, HILDA framework, sampling strategies (RGB Contrast, Embedding Clustering, Logit Uncertainty), CVAT/Label Studio annotation workflows, annotation budgets, or human-in-the-loop (HITL) workflows. Does NOT train models directly — delegates to `herdnet-training` or `wildlife-classification` for the retraining step.

12. **Wildlife skill interactions**: These skills are designed to work together:
    - `megadetector` → `wildlife-classification`: Detection then species classification
    - `practical-cv-wildlife` → `herdnet-training`: PCV foundations then HerdNet implementation
    - `practical-cv-wildlife` → `wildlife-classification`: PCV foundations then species fine-tuning
    - `megadetector` + `herdnet-training`: Compare detection approaches (YOLO wrapper vs point-based)
    - `sahi-inference` + `megadetector`: Tiled MegaDetector inference on large drone images
    - `sahi-inference` vs `herdnet-training`: Compare box-based (SAHI+YOLO) vs heatmap-based (HerdNet Stitcher) tiled inference
    - `active-learning-wildlife` → `herdnet-training`: Active learning loop with HerdNet as the retrained model
    - `active-learning-wildlife` → `wildlife-classification`: Active learning for species classifier improvement

## Key Rules

- All claims must have citations
- Evidence hierarchy respected (meta-analyses > RCTs > cohort > case reports > expert opinion)
- Contradictions disclosed with evidence quality comparison
- AI disclosure in all reports
- Default output language matches user input (Traditional Chinese or English)

## Full Academic Pipeline

```
deep-research (socratic/full)
  → academic-paper (plan/full)
    → academic-paper-reviewer (full/guided)
      → academic-paper (revision)
        → academic-paper-reviewer (re-review, max 2 loops)
          → academic-paper (format-convert → final output)
```

## Handoff Protocol

### deep-research → academic-paper
Materials: RQ Brief, Methodology Blueprint, Annotated Bibliography, Synthesis Report, INSIGHT Collection

### academic-paper → academic-paper-reviewer
Materials: Complete paper text. field_analyst_agent auto-detects domain and configures reviewers.

### academic-paper-reviewer → academic-paper (revision)
Materials: Editorial Decision Letter, Revision Roadmap, Per-reviewer detailed comments

## Version Info
- **Version**: 3.0
- **Last Updated**: 2026-03-13
- **Authors**: Cheng-I Wu (academic skills), Christian Winkelmann (wildlife AI skills)
- **License**: CC-BY-NC 4.0
