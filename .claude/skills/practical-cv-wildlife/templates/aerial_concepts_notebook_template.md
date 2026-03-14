# Aerial Concepts Notebook Template

## Usage

This template defines the cell structure for a Jupyter notebook on aerial imagery fundamentals. The `aerial_concepts_agent` uses this template to generate the complete notebook content.

---

## Notebook Metadata

```
Title: Aerial Imagery Fundamentals for Wildlife Surveys
Module: New (insert after PCV Module 4 or Module 6)
Estimated time: 2-3 hours
Prerequisites: PCV Module 1 (image representation), PCV Module 4 (CNNs)
```

---

## Cell Structure

### Cell 1 [markdown]: Title and Learning Objectives

```markdown
# Aerial Imagery Fundamentals for Wildlife Surveys

In this module, you will learn the physics and mathematics of drone-based imaging for wildlife surveys. Every concept connects back to what you learned in PCV Modules 1 and 4 about image representation and spatial features.

## Learning Objectives

After completing this notebook, you will be able to:
1. Calculate Ground Sampling Distance (GSD) from sensor and flight parameters
2. Determine the image footprint at a given altitude
3. Plan a drone survey with specified overlap requirements
4. Assess motion blur risk for a given flight configuration
5. Explain the orthomosaic generation pipeline and its relevance to wildlife detection
```

### Cell 2 [markdown]: Prerequisites

```markdown
## Prerequisites

- **PCV Module 1:** You understand what a pixel is, how images are stored as tensors, and what image resolution means
- **PCV Module 4:** You understand convolutions, receptive fields, and how spatial features are extracted

**Key connection:** In Module 1, you learned that an image is a grid of pixels. In aerial imaging, each pixel represents a physical area on the ground. The size of that area -- the Ground Sampling Distance -- determines whether you can detect a 35 cm iguana or not.
```

### Cell 3 [code]: Setup

```python
# Setup: Import libraries for aerial imagery calculations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass

# Drone sensor specifications (DJI Mavic 2 Pro)
@dataclass
class DroneSpecs:
    """Camera and sensor specifications for a drone."""
    name: str
    sensor_width_mm: float
    sensor_height_mm: float
    focal_length_mm: float
    image_width_px: int
    image_height_px: int

# Mavic 2 Pro (used in Miesner 2025 thesis)
MAVIC_2_PRO = DroneSpecs(
    name="DJI Mavic 2 Pro",
    sensor_width_mm=13.2,
    sensor_height_mm=8.8,
    focal_length_mm=10.3,
    image_width_px=5472,
    image_height_px=3648,
)

print(f"Loaded specs for: {MAVIC_2_PRO.name}")
print(f"  Sensor: {MAVIC_2_PRO.sensor_width_mm} x {MAVIC_2_PRO.sensor_height_mm} mm")
print(f"  Focal length: {MAVIC_2_PRO.focal_length_mm} mm")
print(f"  Resolution: {MAVIC_2_PRO.image_width_px} x {MAVIC_2_PRO.image_height_px} px")
print(f"  Megapixels: {MAVIC_2_PRO.image_width_px * MAVIC_2_PRO.image_height_px / 1e6:.1f} MP")
```

### Cell 4 [markdown]: Section 1 -- Ground Sampling Distance

```markdown
## Section 1: Ground Sampling Distance (GSD)

**Key concept:** The Ground Sampling Distance (GSD) is the physical distance on the ground represented by one pixel in the image, measured in meters per pixel or centimeters per pixel.

> **PCV Connection:** In Module 1, you learned that an image is a grid of pixels with values 0-255. You thought of resolution as "how many pixels wide." In aerial imaging, resolution has a physical meaning: GSD tells you how many centimeters of ground each pixel captures.

### Formula

$$
\text{GSD} = \frac{S_h \times H}{f \times H_{px}}
$$

Where:
- $S_h$ = sensor height in meters
- $H$ = flight altitude above ground in meters
- $f$ = focal length in meters
- $H_{px}$ = image height in pixels

### Why it matters

A marine iguana is approximately 35 cm long. At GSD = 0.94 cm/px (40 m altitude), the iguana spans ~37 pixels. At GSD = 1.41 cm/px (60 m altitude), it spans only ~25 pixels. This difference directly affects detection accuracy.
```

### Cell 5 [code]: GSD Demonstration

```python
# Demonstration: Calculate GSD for the Mavic 2 Pro at multiple altitudes

def calculate_gsd(sensor_height_mm, altitude_m, focal_length_mm, image_height_px):
    """Calculate Ground Sampling Distance in meters per pixel."""
    sensor_height_m = sensor_height_mm / 1000
    focal_length_m = focal_length_mm / 1000
    gsd_m = (sensor_height_m * altitude_m) / (focal_length_m * image_height_px)
    return gsd_m

# Calculate for thesis altitudes
for alt in [20, 30, 40, 50, 60, 80, 100]:
    gsd = calculate_gsd(
        MAVIC_2_PRO.sensor_height_mm, alt,
        MAVIC_2_PRO.focal_length_mm, MAVIC_2_PRO.image_height_px
    )
    iguana_px = 0.35 / gsd  # 35 cm iguana
    print(f"Altitude {alt:3d}m: GSD = {gsd*100:.2f} cm/px, "
          f"iguana = {iguana_px:.0f} px")
```

### Cell 6 [code]: GSD Visualization

```python
# Visualize: GSD vs altitude (linear relationship)

altitudes = np.arange(10, 121, 5)
gsds = [calculate_gsd(MAVIC_2_PRO.sensor_height_mm, alt,
                       MAVIC_2_PRO.focal_length_mm,
                       MAVIC_2_PRO.image_height_px) * 100
        for alt in altitudes]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: GSD vs altitude
ax1.plot(altitudes, gsds, 'b-', linewidth=2)
ax1.axhline(y=0.94, color='g', linestyle='--', label='40m thesis altitude')
ax1.axhline(y=1.41, color='r', linestyle='--', label='60m thesis altitude')
ax1.set_xlabel('Altitude (m)')
ax1.set_ylabel('GSD (cm/px)')
ax1.set_title('Ground Sampling Distance vs Flight Altitude')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Iguana size in pixels vs altitude
iguana_sizes = [35 / g for g in gsds]  # 35 cm iguana
ax2.plot(altitudes, iguana_sizes, 'r-', linewidth=2)
ax2.axhline(y=10, color='gray', linestyle=':', label='~10px minimum for detection')
ax2.set_xlabel('Altitude (m)')
ax2.set_ylabel('Iguana body length (pixels)')
ax2.set_title('Iguana Size in Image vs Flight Altitude')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Cell 7 [markdown]: GSD Exercise Instructions

```markdown
### Exercise 1: GSD Calculation

You are planning a drone survey using a different drone. Calculate the GSD for the following specifications:

**DJI Phantom 4 Pro:**
- Sensor: 13.2 mm x 8.8 mm (same as Mavic 2 Pro)
- Focal length: 8.8 mm
- Resolution: 5472 x 3648 pixels

**Questions:**
1. Implement `calculate_gsd_general()` that works with any drone specs
2. At what altitude does a 35 cm iguana span exactly 20 pixels? (minimum for reliable detection)
3. Compare the Phantom 4 Pro to the Mavic 2 Pro: which gives better GSD at the same altitude? Why?
```

### Cell 8 [code]: GSD TODO (Level 1)

```python
# TODO Level 1: Calculate GSD for the Phantom 4 Pro at 40m

phantom4_sensor_height_mm = 8.8
phantom4_focal_length_mm = 8.8   # Note: different from Mavic 2 Pro
phantom4_image_height_px = 3648
phantom4_altitude_m = 40

# TODO: Calculate GSD using the formula (1 line)
phantom4_gsd = None  # <-- Replace this line

print(f"Phantom 4 Pro GSD at 40m: {phantom4_gsd*100:.2f} cm/px")
```

### Cell 9 [code]: GSD Validation

```python
# Validation: Check your GSD calculation
assert phantom4_gsd is not None, "phantom4_gsd is None -- implement the calculation"
expected_gsd = (0.0088 * 40) / (0.0088 * 3648)
assert abs(phantom4_gsd - expected_gsd) < 0.0001, \
    f"Expected GSD ~ {expected_gsd:.5f}, got {phantom4_gsd:.5f}"
print(f"Correct! Phantom 4 Pro GSD at 40m = {phantom4_gsd*100:.2f} cm/px")
print(f"Compare: Mavic 2 Pro GSD at 40m = 0.94 cm/px")
print(f"The Phantom 4 Pro has {'worse' if phantom4_gsd > 0.00937 else 'better'} GSD (longer focal length = better GSD)")
```

### Cell 10 [markdown]: Section 2 -- Image Footprint

```markdown
## Section 2: Image Footprint and Survey Planning

**Key concept:** The image footprint is the physical area on the ground captured by a single image.

$$
\text{footprint}_w = \text{GSD} \times W_{px}
$$
$$
\text{footprint}_h = \text{GSD} \times H_{px}
$$

> **PCV Connection:** In Module 1, image dimensions were just width x height in pixels. Now those dimensions, combined with GSD, tell you how much habitat your camera captures per shot.
```

### Cell 11 [code]: Footprint Demonstration

```python
# Demonstration: Calculate footprint and visualize survey coverage

def calculate_footprint(gsd_m, image_width_px, image_height_px):
    """Calculate ground footprint of a single image."""
    return gsd_m * image_width_px, gsd_m * image_height_px

# At 40m altitude
gsd_40 = calculate_gsd(MAVIC_2_PRO.sensor_height_mm, 40,
                        MAVIC_2_PRO.focal_length_mm, MAVIC_2_PRO.image_height_px)
fw, fh = calculate_footprint(gsd_40, MAVIC_2_PRO.image_width_px,
                              MAVIC_2_PRO.image_height_px)

print(f"At 40m altitude:")
print(f"  Footprint: {fw:.1f} m x {fh:.1f} m")
print(f"  Area: {fw * fh:.0f} m^2 = {fw * fh / 10000:.3f} ha")
```

### Cell 12 [code]: Footprint TODO (Level 2)

```python
# TODO Level 2: Calculate number of images for a survey area

def plan_survey(area_width_m, area_length_m, gsd_m,
                image_width_px, image_height_px,
                front_overlap=0.70, side_overlap=0.50):
    """Calculate the number of images needed to cover a survey area.

    Args:
        area_width_m: Width of survey area in meters
        area_length_m: Length of survey area in meters
        gsd_m: Ground sampling distance in meters/pixel
        image_width_px: Image width in pixels
        image_height_px: Image height in pixels
        front_overlap: Front overlap fraction (default 0.70)
        side_overlap: Side overlap fraction (default 0.50)

    Returns:
        dict with 'footprint_w', 'footprint_h', 'trigger_distance',
             'strip_spacing', 'n_strips', 'images_per_strip', 'total_images'
    """
    # TODO: Implement survey planning
    # Step 1: Calculate footprint dimensions
    # Step 2: Calculate trigger distance from front overlap
    # Step 3: Calculate strip spacing from side overlap
    # Step 4: Calculate number of strips (use np.ceil)
    # Step 5: Calculate images per strip (use np.ceil)
    # Step 6: Calculate total images

    raise NotImplementedError("Implement plan_survey")
```

### Cell 13 [code]: Survey Validation

```python
# Validation: Test your survey planning
gsd_40 = calculate_gsd(8.8, 40, 10.3, 3648)
result = plan_survey(500, 300, gsd_40, 5472, 3648, 0.70, 0.50)

assert 'total_images' in result, "Result must include 'total_images'"
assert result['total_images'] > 0, "Total images must be positive"
assert 500 < result['total_images'] < 1000, \
    f"Expected 500-1000 images for 500x300m at 40m, got {result['total_images']}"

print(f"Survey plan for 500m x 300m area at 40m altitude:")
for key, value in result.items():
    print(f"  {key}: {value:.1f}" if isinstance(value, float) else f"  {key}: {value}")
print("Survey planning test passed!")
```

### Cell 14 [markdown]: Section 3 -- Motion Blur

```markdown
## Section 3: Motion Blur Assessment

**Key concept:** During exposure, the drone moves across the ground, causing each pixel to integrate light from a slightly smeared ground area. The blur (in pixels) depends on flight speed, exposure time, and GSD.

$$
\text{blur}_{px} = \frac{\text{ground\_speed} \times \text{exposure\_time}}{\text{GSD}}
$$

A blur of less than 0.5 pixels is negligible. Above 1.0 pixels, fine details (like iguana body outlines) begin to degrade. Above 2.0 pixels, small object detection becomes unreliable.
```

### Cell 15 [code]: Blur Demonstration and TODO (Level 1)

```python
# Demonstration + TODO: Calculate motion blur

def calculate_blur(ground_speed_ms, exposure_s, gsd_m):
    """Calculate motion blur in pixels."""
    return ground_speed_ms * exposure_s / gsd_m

# Demonstration: blur at 40m, 5 m/s, 1/1000s
blur = calculate_blur(5.0, 1/1000, gsd_40)
print(f"At 40m, 5 m/s, 1/1000s: blur = {blur:.2f} px")

# TODO Level 1: What is the maximum safe exposure time (blur = 1.0 px)?
# max_exposure = ???
max_exposure = None  # <-- Replace this line
print(f"Maximum safe exposure: 1/{1/max_exposure:.0f} s")
```

### Cell 16 [markdown]: Section 4 -- Nadir vs Oblique

```markdown
## Section 4: Nadir vs Oblique Views

**Nadir** (straight down): Camera axis perpendicular to ground. GSD is uniform across the image. Required for counting surveys and orthomosaic generation.

**Oblique** (angled): Camera tilted from vertical. GSD varies across the image -- objects farther from the camera appear at lower resolution. Not suitable for systematic counting.

> **Thesis note:** All counting surveys in Miesner (2025) used nadir imagery exclusively. Oblique images were not used because the non-uniform GSD makes consistent counting impossible.
```

### Cell 17 [markdown]: Section 5 -- Orthomosaic Pipeline

```markdown
## Section 5: From Individual Images to Orthomosaic

An orthomosaic is a geometrically corrected, seamless composite of all survey images. The pipeline:

1. **Feature extraction** -- Find distinctive points (SIFT/ORB) in each image
2. **Feature matching** -- Match points between overlapping images
3. **Bundle adjustment** -- Optimize camera positions in 3D space
4. **Dense reconstruction** -- Build a 3D point cloud of the ground surface
5. **Orthorectification** -- Reproject each image to remove perspective distortion
6. **Blending** -- Stitch into a seamless composite

> **Thesis finding:** Pix4D orthomosaics produced ~0.07 higher F1 score than DroneDeploy for iguana detection. Seam line placement matters -- a seam through an iguana creates an artifact.
```

### Cell 18 [code]: Tile Extraction TODO (Level 2)

```python
# TODO Level 2: Implement tile extraction from an orthomosaic

def extract_tiles(image, tile_size, overlap):
    """Extract overlapping tiles from a large image.

    Args:
        image: numpy array of shape (H, W, C)
        tile_size: size of each tile in pixels (square tiles)
        overlap: overlap between adjacent tiles in pixels

    Returns:
        List of tuples: (tile_array, x_origin, y_origin)
    """
    # TODO: Implement tile extraction
    # Step 1: Calculate stride from tile_size and overlap
    # Step 2: Iterate over the image with the calculated stride
    # Step 3: Handle edge cases (tiles at image boundaries)
    # Step 4: Return list of (tile, x_origin, y_origin) tuples

    raise NotImplementedError("Implement extract_tiles")
```

### Cell 19 [code]: Tile Extraction Validation

```python
# Validation: Test tile extraction
test_image = np.random.rand(1000, 1200, 3)
tiles = extract_tiles(test_image, tile_size=256, overlap=64)

assert len(tiles) > 0, "No tiles extracted"
# Check that tiles cover the full image
# ... additional assertions
print(f"Extracted {len(tiles)} tiles from {test_image.shape[1]}x{test_image.shape[0]} image")
print("Tile extraction test passed!")
```

### Cell 20 [markdown]: Section 6 -- Synthesis

```markdown
## Section 6: From GSD to Detection Performance

Let us connect everything back to the case study:

- **Floreana Island:** Flights at 40 m, GSD ~ 0.94 cm/px, iguanas ~37 px -> F1 = 0.934
- **Fernandina Island:** Mixed flights (40 m + 60 m), iguanas 25-37 px -> F1 = 0.843

The higher GSD (lower altitude) on Floreana provides more pixels per iguana, leading to better detection. But lower altitude also means:
- Smaller footprint -> more images needed -> longer flight time
- Greater risk of disturbing animals with drone noise

> **Design question:** If battery life limits you to 25 minutes of flight, how do you balance GSD (detection quality) against coverage area? This is the fundamental trade-off in aerial wildlife surveys.
```

### Cell 21 [markdown]: Solutions

```markdown
## Solutions

Expand the cells below to see solutions for each exercise.
```

### Cell 22 [code]: Solutions

```python
# ---------- SOLUTIONS ----------
# ... complete solutions for all TODO exercises
```

### Cell 23 [markdown]: Summary

```markdown
## Summary

In this module, you learned:
1. **GSD** determines the physical resolution of aerial images -- it is the bridge between pixels and the real world
2. **Footprint** and **overlap** determine survey coverage and image count
3. **Motion blur** limits minimum shutter speed at a given flight speed
4. **Nadir imagery** is required for consistent counting surveys
5. **Orthomosaics** stitch individual images into a single large image for analysis

**Next:** In the Detection module, you will learn how to process these large aerial images to detect and count wildlife.
```
