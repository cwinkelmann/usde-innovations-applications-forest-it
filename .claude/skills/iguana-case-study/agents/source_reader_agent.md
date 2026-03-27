# Source Reader Agent — Iguana Case Study

## Role

Read thesis and defence slide LaTeX sources, extract relevant content for the
requested topic, and produce a structured content brief that the material
generator can turn into course material.

## Source Locations

- **Defence slides**: `/Users/christian/PycharmProjects/hnee/defence_master_thesis/main.tex`
- **Thesis chapters**: `/Users/christian/PycharmProjects/hnee/master_thesis_latex/Subsections/`
  - Introduction: `Intro_2/1_*.tex`
  - Methods: `Methods/2_*.tex`
  - Results: `Results/3_*.tex`
  - Discussion: `4_1_discussion.tex`
  - Conclusion: `4_2_conclusion.tex`
  - Outlook: `4_3_outlook.tex`
- **Thesis tables**: `/Users/christian/PycharmProjects/hnee/master_thesis_latex/tables/`
- **Bibliography**: `/Users/christian/PycharmProjects/hnee/master_thesis_latex/references/`

## Instructions

1. **Read the relevant LaTeX source files** for the user's requested topic.
   Use the topic-to-section mapping from SKILL.md to identify which files to read.

2. **Extract**:
   - Key findings and numerical results (always include exact numbers)
   - Relevant figures with their file paths and what they show
   - Tables with their data
   - Citations that support each claim
   - Experimental methodology relevant to the topic

3. **Output a structured brief** with:
   - `## Topic Summary` — 2-3 paragraph teaching-level explanation
   - `## Key Numbers` — bullet list of quantitative findings
   - `## Available Figures` — table of figure paths + descriptions
   - `## Available Tables` — table data extracted from .tex
   - `## Citations` — relevant references with short descriptions
   - `## Teaching Angles` — 3-5 ways this content could be framed for students

4. **Do NOT generate course material** — that is the material_generator_agent's job.
   Your output is a research brief, not a lecture.

## Topic-to-File Mapping

| Topic keyword | Primary files to read |
|---|---|
| conservation, motivation, why AI | `Intro_2/1_01_conservation_context.tex`, `1_0_introduction.tex` |
| marine iguana, biology, species | `Intro_2/1_02_marine_iguana.tex` |
| monitoring, survey, classical | `Intro_2/1_03_current_monitoring.tex` |
| challenges, camouflage, dense | `Intro_2/1_04_computer_vision_challenges.tex` |
| orthomosaic, pipeline, tiling | `Intro_2/1_30_orthomosaic_workflow.tex` |
| objectives, research question | `Intro_2/1_08_objectives.tex` |
| dataset, Iguanas From Above | `Methods/2_1_2_iguanas_from_above_dataset.tex` |
| HerdNet, architecture, DLA | `Methods/2_1_4_herdnet_architecture.tex` |
| baseline, experiment design | `Methods/2_2_0_experiment_design.tex`, `2_2_1_baseline.tex` |
| training curve, data saturation | `Methods/2_2_2_Training_Curve.tex`, `Results/3_Training_Curve.tex` |
| annotation, head vs body | `Methods/2_2_3_HEAD_vs_Body.tex`, `Results/3_HEAD_vs_Body.tex` |
| hyperparameter, optimization | `Methods/2_2_4_Hyperparameter_Optimisation.tex`, `Results/3_Exp_3_hyperparameter.tex` |
| HITL, human-in-the-loop | `Methods/2_3_0_HumanInTheLoop.tex`, `Results/3_Human_In_The_Loop.tex` |
| results, performance, F1 | `Results/3_0_0_results.tex`, `3_Z5_final_result.tex` |
| human baseline, expert | `Results/3_1_results_human.tex` |
| discussion, limitations | `4_1_discussion.tex` |
| conclusion | `4_2_conclusion.tex` |
| outlook, future work | `4_3_outlook.tex` |
