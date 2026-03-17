---
name: iguana-case-study
description: "Galápagos marine iguana detection case study skill. Creates course material (lectures, practicals, exercises, exam questions) grounded in Winkelmann (2025) master thesis and defence slides. Covers the full pipeline: drone survey design, orthomosaic creation, point-based detection with HerdNet/DLA, annotation protocols (head vs body-center), hyperparameter optimization, human-in-the-loop workflows, cross-island generalization challenges, and image quality dependencies. Source material is the real Iguanas From Above project across five Galápagos islands."
metadata:
  version: "1.0"
  last_updated: "2026-03-17"
  thesis_author: "Christian Winkelmann"
  thesis_doi: "10.6084/m9.figshare.30719999"
---

# Iguana Case Study — Course Material Generator

## Purpose

Generate teaching material for the FIT module grounded in the Galápagos marine
iguana detection case study. All content traces back to real experiments, real
data, and real findings from the master thesis and defence presentation.

## Source Material Locations

| Source | Path | Format |
|--------|------|--------|
| Defence slides (primary) | `/Users/christian/PycharmProjects/hnee/defence_master_thesis/main.tex` | Beamer LaTeX |
| Defence figures | `/Users/christian/PycharmProjects/hnee/defence_master_thesis/figures/` | PNG/PDF |
| Thesis LaTeX (full) | `/Users/christian/PycharmProjects/hnee/master_thesis_latex/` | LaTeX chapters |
| Thesis chapters | `/Users/christian/PycharmProjects/hnee/master_thesis_latex/Subsections/` | .tex files |
| Thesis figures | `/Users/christian/PycharmProjects/hnee/master_thesis_latex/figures/` | PNG/PDF |
| Thesis tables | `/Users/christian/PycharmProjects/hnee/master_thesis_latex/tables/` | .tex tables |
| Thesis bibliography | `/Users/christian/PycharmProjects/hnee/master_thesis_latex/references/` | .bib files |
| Published thesis PDF | DOI: [10.6084/m9.figshare.30719999](https://doi.org/10.6084/m9.figshare.30719999) | PDF |
| HerdNet fork | [github.com/cwinkelmann/HerdNet](https://github.com/cwinkelmann/HerdNet) | Python |

## Trigger Keywords

iguana case study, marine iguana, Galápagos detection, Iguanas From Above, iguana
slides, thesis material, case study lecture, case study exercise, exam question
iguana, iguana practical, drone iguana survey, Winkelmann thesis, defence slides,
create lecture from thesis, course material from research, iguana annotation,
body-center annotation, head vs body annotation, cross-island generalization,
iguana HITL, iguana human-in-the-loop

## Does NOT Trigger

| If the user wants... | Use this skill instead |
|---|---|
| Train HerdNet from scratch (code-focused) | herdnet-training |
| MegaDetector on camera traps | megadetector |
| Species classification fine-tuning | wildlife-classification |
| SAHI tiled inference code | sahi-inference |
| Active learning loop design | active-learning-wildlife |
| PCV course gap analysis | practical-cv-wildlife |

## Agent Team

| Agent | Role | Phase |
|---|---|---|
| `source_reader_agent` | Reads thesis/slides LaTeX, extracts relevant content, figures, and tables for the requested topic | Research |
| `material_generator_agent` | Produces course material (lecture notes, Marimo notebooks, exercises, exam questions) from extracted content | Generation |

## Orchestration

```
User Input (e.g., "create a lecture on annotation protocols using my thesis")
    |
=== Phase 0: INTAKE ===
    |
    Determine mode + topic from user request
    |
=== Phase 1: SOURCE EXTRACTION ===
    |
    [source_reader_agent]
    - Read relevant thesis chapters / slide sections
    - Extract key findings, figures, tables, citations
    - Identify which thesis experiments map to the topic
    Output: Structured content brief with citations
    |
=== Phase 2: MATERIAL GENERATION ===
    |
    [material_generator_agent]
    - Generate requested material type
    - Ground all claims in thesis findings
    - Reference figure paths for slide inclusion
    - Follow FIT module conventions (Marimo format, exercise structure)
    Output: Course material file(s)
```

## Operational Modes

| Mode | Description | Agents Active |
|---|---|---|
| `create-lecture` | Generate lecture notes or slide outline for a topic | source_reader → material_generator |
| `create-practical` | Generate a Marimo notebook practical grounded in thesis data | source_reader → material_generator |
| `create-exercise` | Generate student exercises with solutions from thesis findings | source_reader → material_generator |
| `create-exam` | Generate exam questions (multiple choice, short answer, discussion) | source_reader → material_generator |
| `extract-figures` | List available figures for a topic with paths and descriptions | source_reader |
| `summarize-topic` | Explain a thesis topic at teaching level (no material generation) | source_reader |

## Case Study Knowledge Base

### Thesis Structure → Teaching Topics

| Thesis Section | Teaching Topic | Key Takeaway for Students |
|---|---|---|
| 1.01 Conservation context | Why AI in ecology | Traditional methods can't scale; automation is necessary |
| 1.02 Marine iguana biology | Study species background | Colonial behavior + basking = drone-observable |
| 1.03 Current monitoring | Classical vs AI methods | Ground surveys miss 17-35% of animals |
| 1.04 CV challenges | Real-world detection challenges | Camouflage, dense aggregations, variable image quality |
| 1.30 Orthomosaic workflow | Data pipeline | Raw frames → orthomosaic → tiles → annotations |
| 2.1.4 HerdNet architecture | Point-based detection | FIDT > Gaussian kernels for dense colonies |
| 2.2.1 Baseline | Model evaluation basics | F1, precision, recall on held-out test sets |
| 2.2.2 Training curve | Data saturation | More data has diminishing returns; quality > quantity |
| 2.2.3 HEAD vs BODY | Annotation protocol design | Body-center annotations outperform head (+0.17 F1) |
| 2.2.4 Hyperparameters | Optimization strategy | Matching threshold, LMDS kernel, augmentation tuning |
| 2.3.0 HITL | Human-in-the-loop | 330 annotations/hour; model sometimes exceeds humans |
| 3.Z5 Final results | Performance summary | F1 = 0.85-0.93 across islands |
| 4.1 Discussion | Challenges & limitations | No true ground truth; island-specific models needed |
| 4.3 Outlook | Future directions | Automated flights, onboard inference, geo-tracking |

### Key Findings (for grounding course material)

1. **Annotation quality > quantity**: Learning curve plateaus at ~950 annotations (Floreana) / ~3,100 (Fernandina). More data doesn't help; better annotation protocols do.

2. **Body-center >> head annotations**: +0.17 F1 improvement. HerdNet learns center-of-mass features via FIDT heatmaps; head orientation varies with posture.

3. **Smaller models win**: DLA-34 outperforms DLA-60/102/169. Larger models overfit to ~30% annotation noise. Confirms Nakkiran et al. on label noise sensitivity.

4. **Geometric augmentation helps, color doesn't**: RandomRotate90, HorizontalFlip, Perspective are beneficial. HSV augmentation degrades performance (volcanic rock texture matters more than color).

5. **Cross-island training fails**: Mixing islands degrades per-island F1. Visual heterogeneity too large. Recommend island-specific models.

6. **Image quality is the bottleneck**: GSD, orthomosaic software (Pix4D > DroneDeploy), motion blur, ISO noise all substantially impact detection. Not a pure ML problem.

7. **Experts are imperfect**: 4 experts, 496 images, 1,014 consensus iguanas. Experts overcount by 3-8% but also miss objects. No true ground truth exists.

8. **HITL efficiency**: 330 annotations/hour on Fernandina (11,500 predictions corrected in 35h). Model sometimes detects more than humans at low-density sites.

### Available Figures by Topic

When generating material, reference these figure directories:

| Topic | Figure Location (defence) | Figure Location (thesis) |
|---|---|---|
| Study area maps | `defence/figures/old/` | `thesis/figures/maps/` |
| HerdNet architecture | `defence/figures/thesis/` | `thesis/figures/neural_network/` |
| FIDT vs Gaussian | `defence/figures/thesis/` | `thesis/figures/neural_network/` |
| Detection examples | `defence/figures/thesis/` | `thesis/figures/detection_examples/` |
| Training curves | `defence/figures/thesis/` | `thesis/figures/` |
| Hyperparameter sweeps | `defence/figures/thesis/` | `thesis/figures/` |
| Flight statistics | — | `thesis/figures/drone_fernandina_cabo_douglas/` |
| Annotation workflow | `defence/figures/old/` | `thesis/figures/` |

### Key Citations (from thesis bibliography)

| Short cite | Full reference | Relevance |
|---|---|---|
| Delplanque et al. (2023) | HerdNet paper | Core model architecture |
| Liang et al. (2023) | FIDT paper | Density map approach |
| Varela-Jaramillo et al. (2023) | Iguana drone pilot study | Drone vs ground counts |
| Varela-Jaramillo et al. (2025) | Citizen science counts | Zooniverse validation |
| Eikelboom et al. (2019) | Aerial object detection | Reference dataset |
| Nakkiran et al. (2021) | Double descent / label noise | Explains DLA-34 > DLA-169 |
| Stock et al. (2023) | Spatial data leakage | Critical for evaluation |

## Course Material Conventions

When generating material, follow these rules (from CLAUDE.md):

1. **Marimo notebooks**: All practicals are `.py` files with `marimo.App()` structure
2. **Cell structure**: Context → Script snippet → Exercise → Reflection
3. **No training from scratch**: Focus on tool fluency, not backpropagation
4. **Ground in real data**: Always reference Iguanas From Above experiments
5. **Honest about limitations**: Discuss what fails (Genovesa F1=0.43, cross-island)
6. **Windows compatibility**: Use `os.path`, avoid bash-specific syntax
