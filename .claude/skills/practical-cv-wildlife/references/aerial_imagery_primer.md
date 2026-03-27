# Aerial Imagery Primer -- GSD, Footprint, Overlap, and Motion Blur

## Purpose

This is the primary reference for all aerial imagery calculations used in the practical-cv-wildlife skill. It provides formula derivations, worked examples with the DJI Mavic 2 Pro (thesis drone), and generalizable formulas for any drone platform.

Used by: `aerial_concepts_agent`, `exercise_generator_agent`

---

## Sensor Parameters: DJI Mavic 2 Pro

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Sensor width | S_w | 13.2 | mm |
| Sensor height | S_h | 8.8 | mm |
| Focal length | f | 10.3 | mm |
| Image width | W_px | 5472 | pixels |
| Image height | H_px | 3648 | pixels |
| Total resolution | - | 19,961,856 | pixels (~20 MP) |
| Pixel pitch (width) | p_w | 13.2 / 5472 = 0.002412 | mm/px |
| Pixel pitch (height) | p_h | 8.8 / 3648 = 0.002412 | mm/px |
| Sensor aspect ratio | - | 3:2 | - |

---

## Ground Sampling Distance (GSD)

### Formula Derivation

GSD is derived from similar triangles formed by the camera's pinhole model:

```
Camera Sensor Plane                Ground Plane
+------------------+               +-----------------------------+
|                  |               |                             |
|    S_h (sensor   |               |                             |
|    height)       |               |   Footprint height          |
|                  |               |                             |
+------------------+               +-----------------------------+
         |                                      |
         |<-- f (focal length) -->|              |
         |                        |<-- H (altitude) ----------->|
```

By similar triangles:
```
S_h / f = Footprint_h / H
```

Therefore:
```
Footprint_h = (S_h * H) / f
```

GSD is the ground distance per pixel:
```
GSD = Footprint_h / H_px = (S_h * H) / (f * H_px)
```

### General Formula

```
GSD = (S_h * H) / (f * H_px)
```

Where:
- `S_h` = sensor height (meters) -- use consistent units
- `H` = flight altitude AGL (meters)
- `f` = focal length (meters) -- use consistent units
- `H_px` = image height in pixels

Alternatively, using pixel pitch:
```
GSD = p_h * H / f
```

Where `p_h` = sensor height / image height in pixels (in meters).

### Worked Examples

**At 40 m altitude (Mavic 2 Pro):**
```
GSD = (0.0088 m * 40 m) / (0.0103 m * 3648 px)
    = 0.352 / 37.5744
    = 0.009370 m/px
    = 0.937 cm/px
```

**At 60 m altitude (Mavic 2 Pro):**
```
GSD = (0.0088 m * 60 m) / (0.0103 m * 3648 px)
    = 0.528 / 37.5744
    = 0.014055 m/px
    = 1.406 cm/px
```

**At 80 m altitude (Mavic 2 Pro):**
```
GSD = (0.0088 m * 80 m) / (0.0103 m * 3648 px)
    = 0.704 / 37.5744
    = 0.018740 m/px
    = 1.874 cm/px
```

### GSD Quick Reference Table (Mavic 2 Pro)

| Altitude (m) | GSD (cm/px) | Iguana body (~35 cm) in pixels | Iguana head (~5 cm) in pixels |
|---------------|-------------|-------------------------------|------------------------------|
| 20 | 0.47 | ~74 px | ~11 px |
| 30 | 0.70 | ~50 px | ~7 px |
| 40 | 0.94 | ~37 px | ~5 px |
| 50 | 1.17 | ~30 px | ~4 px |
| 60 | 1.41 | ~25 px | ~4 px |
| 80 | 1.87 | ~19 px | ~3 px |
| 100 | 2.34 | ~15 px | ~2 px |

**Thesis note:** Body-center annotations outperform head annotations by ~0.10 F1. At 60 m, iguana heads are only ~4 pixels -- too small for reliable annotation and detection.

---

## Image Footprint

### Formulas

```
Footprint_w = GSD * W_px     (width of ground area captured)
Footprint_h = GSD * H_px     (height of ground area captured)
Footprint_area = Footprint_w * Footprint_h
```

### Worked Examples

**At 40 m:**
```
Footprint_w = 0.009370 * 5472 = 51.27 m
Footprint_h = 0.009370 * 3648 = 34.18 m
Footprint_area = 51.27 * 34.18 = 1752.4 m^2 = 0.175 ha
```

**At 60 m:**
```
Footprint_w = 0.014055 * 5472 = 76.91 m
Footprint_h = 0.014055 * 3648 = 51.27 m
Footprint_area = 76.91 * 51.27 = 3943.2 m^2 = 0.394 ha
```

---

## Overlap Calculation

### Front (Forward) Overlap

Front overlap determines the distance between consecutive photo triggers along the flight line.

```
trigger_distance = Footprint_h * (1 - overlap_front)
```

**At 40 m, 70% front overlap (thesis value):**
```
trigger_distance = 34.18 * (1 - 0.70) = 34.18 * 0.30 = 10.25 m
```

At 5 m/s flight speed:
```
trigger_interval = 10.25 / 5.0 = 2.05 seconds
```

### Side (Lateral) Overlap

Side overlap determines the spacing between adjacent flight lines (strips).

```
strip_spacing = Footprint_w * (1 - overlap_side)
```

**At 40 m, 50% side overlap (thesis value):**
```
strip_spacing = 51.27 * (1 - 0.50) = 51.27 * 0.50 = 25.64 m
```

### Number of Images for a Survey Area

```
n_strips = ceil(area_width / strip_spacing) + 1
images_per_strip = ceil(area_length / trigger_distance) + 1
total_images = n_strips * images_per_strip
```

**For a 500 m x 300 m area at 40 m, 70/50 overlap:**
```
n_strips = ceil(300 / 25.64) + 1 = 12 + 1 = 13
images_per_strip = ceil(500 / 10.25) + 1 = 49 + 1 = 50
total_images = 13 * 50 = 650 images
```

### Number of Overlapping Images per Ground Point

```
images_per_point_along_track = 1 / (1 - overlap_front)
images_per_point_across_track = 1 / (1 - overlap_side)
total_images_per_point = images_per_point_along_track * images_per_point_across_track
```

**At 70/50 overlap:**
```
along_track = 1 / 0.30 = 3.33
across_track = 1 / 0.50 = 2.00
total = 3.33 * 2.00 = 6.67
```

Each ground point appears in approximately 6-7 images, which is sufficient for Structure from Motion reconstruction.

---

## Motion Blur Model

### Formula

```
blur_px = (ground_speed * exposure_time) / GSD
```

Where:
- `ground_speed` = drone speed over ground (m/s)
- `exposure_time` = camera shutter speed (seconds)
- `GSD` = ground sampling distance (m/px)

Result is blur measured in pixels. Blur < 1.0 px is negligible. Blur > 1.0 px degrades feature edges. Blur > 2.0 px significantly impacts small object detection.

### Worked Examples

**At 40 m altitude, 5 m/s, various shutter speeds:**

| Shutter Speed | Exposure (s) | Blur (px) | Impact |
|---------------|-------------|-----------|--------|
| 1/2000 | 0.0005 | 0.27 | Negligible |
| 1/1000 | 0.001 | 0.53 | Negligible |
| 1/500 | 0.002 | 1.07 | Edge degradation |
| 1/250 | 0.004 | 2.13 | Significant |
| 1/125 | 0.008 | 4.27 | Unacceptable |

**At 60 m altitude, 5 m/s, various shutter speeds:**

| Shutter Speed | Exposure (s) | Blur (px) | Impact |
|---------------|-------------|-----------|--------|
| 1/2000 | 0.0005 | 0.18 | Negligible |
| 1/1000 | 0.001 | 0.36 | Negligible |
| 1/500 | 0.002 | 0.71 | Negligible |
| 1/250 | 0.004 | 1.42 | Edge degradation |
| 1/125 | 0.008 | 2.85 | Significant |

**Observation:** Higher altitude = larger GSD = less blur in pixels (but also less detail per animal). This is a trade-off: flying higher reduces blur risk but reduces animal pixel count.

### Maximum Safe Exposure Time

```
max_exposure = GSD / ground_speed     (for blur = 1.0 px threshold)
```

**At 40 m, 5 m/s:**
```
max_exposure = 0.00937 / 5.0 = 0.001874 s ~ 1/533 s
```

Recommendation: Use at least 1/1000 s at 40 m altitude and 5 m/s flight speed.

---

## Nadir vs Oblique Views

### Nadir (Straight Down)

- Camera angle: 90 degrees to ground (0 degrees from vertical)
- GSD: Uniform across the image (ignoring lens distortion)
- Use case: Systematic counting surveys, orthomosaic generation
- Thesis usage: All counting surveys used nadir imagery

### Oblique (Angled)

- Camera angle: Typically 30-60 degrees from vertical
- GSD: Varies across the image (smaller near horizon, larger at bottom)
- Use case: Visual inspection, 3D reconstruction
- Thesis usage: Not used for counting

### GSD Variation in Oblique Images

For an oblique image at angle theta from nadir:
```
GSD_near = GSD_nadir / cos(theta)    (at image bottom, closest to drone)
GSD_far = GSD_nadir / cos(theta)^2   (at image top, farthest from drone)
```

This non-uniform GSD makes oblique images unsuitable for consistent counting: animals at different distances appear at different scales.

---

## Orthorectification Pipeline

### Steps

```
1. Individual Images (with GPS/EXIF metadata)
   |
2. Feature Extraction (SIFT, ORB, or SuperPoint)
   | - Identify keypoints in each image
   |
3. Feature Matching
   | - Match keypoints between overlapping images
   | - RANSAC for outlier rejection
   |
4. Bundle Adjustment
   | - Optimize camera poses (position + orientation)
   | - Minimize reprojection error across all images
   |
5. Dense Point Cloud Generation
   | - Multi-view stereo matching
   | - Produces colored 3D point cloud
   |
6. Digital Surface Model (DSM)
   | - Interpolate point cloud to regular grid
   | - Records elevation at each grid cell
   |
7. Orthorectification
   | - Reproject each image pixel to ground plane using DSM
   | - Removes perspective distortion
   |
8. Orthomosaic
   | - Blend orthorectified images into seamless composite
   | - Seam line optimization to minimize visible joints
```

### Software Comparison (from Thesis)

| Software | Type | Quality | Speed | Cost | Thesis Finding |
|----------|------|---------|-------|------|----------------|
| Pix4D | Desktop + cloud | HIGH | Medium | Paid ($350/mo) | F1 ~0.07 higher than DroneDeploy |
| DroneDeploy | Cloud-only | Medium | Fast | Paid ($499/mo) | Lower quality seam lines |
| OpenDroneMap | Open-source desktop | Variable | Slow | Free | Not tested in thesis |
| Agisoft Metashape | Desktop | HIGH | Medium | Paid ($179 one-time) | Common academic choice |

**Thesis recommendation:** Pix4D produces higher quality orthomosaics that lead to better detection performance. The F1 delta of ~0.07 between Pix4D and DroneDeploy orthomosaics is significant.
