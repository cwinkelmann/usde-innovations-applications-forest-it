# Exercise Design Patterns -- PCV Notebook Conventions

## Purpose

This reference documents the exact structure, conventions, and patterns used in PCV course notebooks. All new exercises created by the `exercise_generator_agent` and all adapted notebooks by the `wildlife_adapter_agent` must follow these patterns to maintain course consistency.

---

## PCV Notebook Cell Structure

### Standard Cell Ordering

Every PCV notebook follows this sequence:

```
[1] markdown: Title + Description
[2] markdown: Learning Objectives
[3] markdown: Prerequisites
[4] code:     Setup (imports, configuration, utility functions)
[5] markdown: Section 1 Theory
[6] code:     Section 1 Demonstration (instructor code)
[7] markdown: Section 1 Exercise Instructions
[8] code:     Section 1 TODO Scaffold
[9] code:     Section 1 Validation / Assertions
[10-14]: Repeat pattern for Section 2
[15-19]: Repeat pattern for Section 3
[N] markdown: Summary / Key Takeaways
[N+1] markdown: Further Reading / References
```

### Cell Type Conventions

**Markdown cells:**
- Use `#` for notebook title (only one `#` per notebook)
- Use `##` for section headers
- Use `###` for subsections within a section
- Use bold `**text**` for key terms on first use
- Use inline code backticks for variable names, function names, and file paths
- Use blockquotes `>` for tips, warnings, or important notes

**Code cells:**
- Maximum 30 lines per cell (break longer code into multiple cells)
- Every code cell has a purpose comment as the first line: `# Train the model for 10 epochs`
- Import cells are always the first code cell
- Print statements for intermediate results (students should see output as they go)
- No cell should take more than 60 seconds to run (except explicit training cells)

---

## Learning Objectives Pattern

```markdown
## Learning Objectives

After completing this notebook, you will be able to:
1. [Action verb] [specific skill] (e.g., "Calculate Ground Sampling Distance from sensor parameters")
2. [Action verb] [specific skill]
3. [Action verb] [specific skill]
4. [Action verb] [specific skill] (optional, max 5)
```

**Action verbs used in PCV (Bloom's taxonomy):**
- Remember: Define, List, Identify
- Understand: Explain, Describe, Compare
- Apply: Calculate, Implement, Use
- Analyze: Analyze, Differentiate, Examine
- Evaluate: Evaluate, Assess, Compare
- Create: Design, Build, Implement

---

## Prerequisites Pattern

```markdown
## Prerequisites

Before starting this notebook, you should have completed:
- **Module N: [Module Title]** -- specifically: [concept needed]
- **Notebook: [Notebook Name]** -- you will use [function/concept] from this notebook

**Required packages:**
```python
# These should already be installed from Module N setup
# pip install torch torchvision matplotlib numpy
```
```

---

## Setup Cell Pattern

```python
# Setup: Install dependencies and configure environment
# Run this cell first -- it may take a minute to download data

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
SEED = 42
torch.manual_seed(SEED)

print(f"Using device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")
```

**Conventions:**
- All imports in one cell
- Device detection with informative print
- Data directory creation
- Random seed for reproducibility
- Version print for debugging

---

## Theory Cell Pattern

```markdown
## Section N: [Topic Name]

**Key concept:** [one-sentence definition]

[2-3 paragraphs of explanation, building from what students already know]

> **PCV Connection:** In Module X, you learned [concept]. Here we extend this to [wildlife context].

**Formula:**

$$
\text{GSD} = \frac{S_h \times H}{f \times H_{px}}
$$

Where:
- $S_h$ = sensor height (m)
- $H$ = flight altitude (m)
- $f$ = focal length (m)
- $H_{px}$ = image height (pixels)
```

**Conventions:**
- Start with a bold key concept definition
- Build from known PCV concepts (reference explicitly)
- Use LaTeX for formulas (Jupyter renders it)
- Define all variables after the formula
- Keep theory cells under 300 words

---

## Demonstration Cell Pattern

```python
# Demonstration: Calculate GSD for the Mavic 2 Pro at 40m altitude

# Mavic 2 Pro sensor specifications
sensor_height_mm = 8.8       # mm
focal_length_mm = 10.3       # mm
image_height_px = 3648       # pixels
altitude_m = 40              # meters

# Convert to consistent units (meters)
sensor_height_m = sensor_height_mm / 1000
focal_length_m = focal_length_mm / 1000

# Calculate GSD
gsd = (sensor_height_m * altitude_m) / (focal_length_m * image_height_px)

print(f"At {altitude_m}m altitude:")
print(f"  GSD = {gsd:.5f} m/px = {gsd * 100:.2f} cm/px")
print(f"  A 35cm iguana spans {0.35 / gsd:.0f} pixels")
```

**Conventions:**
- Comment on first line describes what the cell demonstrates
- Variables named descriptively with units in the name or comments
- Step-by-step computation (not one-line formulas)
- Print results with clear labels and units
- Connect result to wildlife context (iguana size in pixels)

---

## TODO Scaffold Pattern

### Level 1: Guided (Fill in 1-2 lines)

```python
# TODO: Calculate GSD for different altitudes
# Use the formula: GSD = (sensor_height * altitude) / (focal_length * image_height_px)

def calculate_gsd(sensor_height_mm, altitude_m, focal_length_mm, image_height_px):
    """Calculate Ground Sampling Distance in meters per pixel.

    Args:
        sensor_height_mm: Physical sensor height in millimeters
        altitude_m: Flight altitude above ground in meters
        focal_length_mm: Lens focal length in millimeters
        image_height_px: Image height in pixels

    Returns:
        gsd_m: Ground Sampling Distance in meters per pixel
    """
    # Convert units from mm to meters
    sensor_height_m = sensor_height_mm / 1000
    focal_length_m = focal_length_mm / 1000

    # TODO: Calculate GSD using the formula above (1 line)
    gsd_m = None  # <-- Replace this line

    return gsd_m
```

### Level 2: Scaffolded (Implement a function with docstring)

```python
# TODO: Implement a function that plans a drone survey

def plan_survey(area_width_m, area_length_m, altitude_m, front_overlap, side_overlap,
                sensor_height_mm=8.8, focal_length_mm=10.3,
                image_width_px=5472, image_height_px=3648):
    """Plan a drone survey and return required number of images.

    Args:
        area_width_m: Survey area width in meters
        area_length_m: Survey area length in meters
        altitude_m: Flight altitude in meters
        front_overlap: Front overlap fraction (e.g., 0.70)
        side_overlap: Side overlap fraction (e.g., 0.50)
        sensor_height_mm: Sensor height in mm (default: Mavic 2 Pro)
        focal_length_mm: Focal length in mm (default: Mavic 2 Pro)
        image_width_px: Image width in pixels (default: Mavic 2 Pro)
        image_height_px: Image height in pixels (default: Mavic 2 Pro)

    Returns:
        dict with keys: 'gsd_m', 'footprint_w_m', 'footprint_h_m',
                        'n_strips', 'images_per_strip', 'total_images'
    """
    # TODO: Implement this function
    # Step 1: Calculate GSD
    # Step 2: Calculate footprint width and height
    # Step 3: Calculate trigger distance (front overlap)
    # Step 4: Calculate strip spacing (side overlap)
    # Step 5: Calculate number of strips and images per strip
    # Step 6: Return results dictionary

    raise NotImplementedError("Implement plan_survey")
```

### Level 3: Open-Ended (Design and implement from specification)

```python
# TODO: Design a complete flight assessment function
#
# Your function should:
# 1. Take drone specs, flight parameters, and environmental conditions as input
# 2. Calculate GSD, footprint, number of images, and flight time
# 3. Assess motion blur risk given flight speed and minimum shutter speed
# 4. Warn if GSD is too coarse for the target animal size
# 5. Return a structured report (dictionary or formatted string)
#
# Consider these real-world constraints:
# - Battery life limits total flight time (~25 min for Mavic 2 Pro)
# - Wind speed affects ground speed (add/subtract from planned speed)
# - Regulatory altitude limits (typically 120m AGL)
#
# There is no single correct solution. Design your function signature
# and implement it.

def assess_flight_plan(...):  # Define your own parameters
    """Your docstring here."""
    # Your implementation here
    pass
```

---

## Assertion / Validation Cell Pattern

```python
# Validation: Test your GSD calculation

# Test 1: Known values from Mavic 2 Pro at 40m
gsd_40 = calculate_gsd(8.8, 40, 10.3, 3648)
assert gsd_40 is not None, "calculate_gsd returned None -- did you implement it?"
assert isinstance(gsd_40, float), f"Expected float, got {type(gsd_40)}"
assert abs(gsd_40 - 0.00937) < 0.001, \
    f"GSD at 40m should be ~0.00937 m/px, got {gsd_40:.5f}"
print(f"Test 1 passed: GSD at 40m = {gsd_40*100:.2f} cm/px")

# Test 2: Known values at 60m
gsd_60 = calculate_gsd(8.8, 60, 10.3, 3648)
assert abs(gsd_60 - 0.01405) < 0.001, \
    f"GSD at 60m should be ~0.01405 m/px, got {gsd_60:.5f}"
print(f"Test 2 passed: GSD at 60m = {gsd_60*100:.2f} cm/px")

# Test 3: Proportionality check (GSD scales linearly with altitude)
ratio = gsd_60 / gsd_40
assert abs(ratio - 1.5) < 0.01, \
    f"GSD ratio (60m/40m) should be 1.5, got {ratio:.2f}"
print(f"Test 3 passed: GSD ratio = {ratio:.2f}")

print("\nAll GSD tests passed!")
```

**Conventions:**
- Test for None first (catches unimplemented functions)
- Test for correct type
- Test for correct value with tolerance
- Include informative error messages with actual vs expected values
- Print success message per test
- Final "All tests passed!" message
- At least 2-3 test cases per TODO

---

## Solution Cell Pattern

```python
# ---------- SOLUTION ----------
# Expand this cell to see the solution. Try to solve it yourself first!

def calculate_gsd(sensor_height_mm, altitude_m, focal_length_mm, image_height_px):
    """Calculate Ground Sampling Distance in meters per pixel."""
    sensor_height_m = sensor_height_mm / 1000
    focal_length_m = focal_length_mm / 1000
    gsd_m = (sensor_height_m * altitude_m) / (focal_length_m * image_height_px)
    return gsd_m

# Verify solution
print(f"GSD at 40m: {calculate_gsd(8.8, 40, 10.3, 3648)*100:.2f} cm/px")
print(f"GSD at 60m: {calculate_gsd(8.8, 60, 10.3, 3648)*100:.2f} cm/px")
```

**Conventions:**
- Clear separator line `# ---------- SOLUTION ----------`
- Warning to try first
- Complete, runnable solution
- Verification output
- In actual Jupyter notebooks, this cell would be collapsed or tagged for hiding

---

## Progressive Difficulty Guidelines

| Level | Scaffolding | Student Writes | Cognitive Load |
|-------|-------------|---------------|----------------|
| 1 -- Guided | Function signature, docstring, all steps except 1-2 lines | 1-2 lines of code | Low (apply formula) |
| 2 -- Scaffolded | Function signature, docstring, step comments | Full function body (10-20 lines) | Medium (combine concepts) |
| 3 -- Open-ended | Problem description only | Function design + implementation | High (design + implement) |

**Transition cues:**
- Level 1 -> 2: "Now that you can calculate GSD, combine it with footprint estimation..."
- Level 2 -> 3: "You have built the components. Now design a complete system..."

---

## Time Estimates by Exercise Type

| Exercise Type | Level 1 | Level 2 | Level 3 | Total |
|---------------|---------|---------|---------|-------|
| Formula implementation (GSD) | 10 min | 20 min | 30 min | 60 min |
| Data pipeline (DataLoader) | 15 min | 30 min | 30 min | 75 min |
| Model training (YOLOv8) | 15 min | 45 min | 45 min | 105 min |
| Evaluation (metrics) | 10 min | 25 min | 25 min | 60 min |
| Visualization (embeddings) | 15 min | 30 min | 30 min | 75 min |

**Buffer:** Add 20% for setup, debugging, and environment issues.
