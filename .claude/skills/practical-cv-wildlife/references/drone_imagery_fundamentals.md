# Drone Imagery Fundamentals -- Complete Reference

## Purpose

This reference provides a comprehensive treatment of drone-based imaging for wildlife surveys. It covers camera geometry, GSD derivation, footprint calculation, overlap planning, motion blur modeling, and orthorectification. It is more detailed than `aerial_imagery_primer.md` and serves as the authoritative source for all drone imaging formulas.

Used by: `aerial_concepts_agent`, `exercise_generator_agent`

---

## Camera Geometry

### Pinhole Camera Model

The pinhole model describes the relationship between a 3D scene point and its 2D projection on the sensor:

```
Scene Point P(X, Y, Z)
      |
      |  (distance Z from camera = flight altitude H)
      |
      v
Camera Center (optical center)
      |
      |  (distance f = focal length)
      |
      v
Sensor Point p(x, y)
```

By similar triangles:
```
x / f = X / Z    =>    x = f * X / Z
y / f = Y / Z    =>    y = f * Y / Z
```

In photogrammetric convention (Z = altitude H, X/Y = ground coordinates):
```
x = f * X / H
y = f * Y / H
```

### Sensor Coordinate System

```
+----------------------------------+
|                                  |
|  (0,0)                          |
|    +---> u (pixel columns)      |
|    |                             |
|    v                             |
|    v (pixel rows)                |
|                                  |
|                     (W_px, H_px) |
+----------------------------------+

Sensor physical dimensions: S_w x S_h (mm)
Image dimensions: W_px x H_px (pixels)
```

### Pixel Pitch

Pixel pitch is the physical size of one pixel on the sensor:

```
pixel_pitch_w = S_w / W_px    (mm/pixel, along width)
pixel_pitch_h = S_h / H_px    (mm/pixel, along height)
```

For square pixels (typical): pixel_pitch_w = pixel_pitch_h

**Mavic 2 Pro:**
```
pixel_pitch = 13.2 / 5472 = 0.002412 mm/px = 2.412 um
```

---

## Ground Sampling Distance (GSD)

### Derivation from First Principles

Consider a single pixel on the sensor. Its physical size on the sensor is `pixel_pitch` (in mm). By the pinhole camera model, this pixel images a ground area of:

```
ground_per_pixel = pixel_pitch * H / f
```

Where:
- `pixel_pitch` = physical size of one pixel on sensor (mm)
- `H` = flight altitude above ground (mm, or use consistent units)
- `f` = focal length (mm)

This gives GSD in the same units as H (typically meters):

```
GSD = (pixel_pitch_mm / 1000) * H_m / (f_mm / 1000)
    = pixel_pitch_mm * H_m / f_mm
```

Or equivalently, using sensor dimensions directly:

```
GSD = (S_h * H) / (f * H_px)
```

This is the standard GSD formula used throughout this skill.

### GSD Along Width vs Height

For non-square sensors (or when computing along different axes):

```
GSD_w = (S_w * H) / (f * W_px)    (along image width)
GSD_h = (S_h * H) / (f * H_px)    (along image height)
```

For the Mavic 2 Pro, since pixel pitch is square (2.412 um both ways), GSD_w = GSD_h.

### GSD vs Altitude (Linear Relationship)

```
GSD(H) = k * H    where k = S_h / (f * H_px)
```

For Mavic 2 Pro:
```
k = 0.0088 / (0.0103 * 3648) = 0.0002342 (m/px per meter of altitude)
k = 0.02342 (cm/px per meter of altitude)
```

So: GSD doubles when altitude doubles. This linear relationship is useful for quick mental estimates.

### Reference Table

| Altitude (m) | GSD (m/px) | GSD (cm/px) | 35 cm iguana (px) | 5 cm head (px) |
|---------------|-----------|-------------|-------------------|----------------|
| 10 | 0.00234 | 0.23 | 150 | 21 |
| 20 | 0.00469 | 0.47 | 75 | 11 |
| 30 | 0.00703 | 0.70 | 50 | 7 |
| 40 | 0.00937 | 0.94 | 37 | 5 |
| 50 | 0.01171 | 1.17 | 30 | 4 |
| 60 | 0.01406 | 1.41 | 25 | 4 |
| 80 | 0.01874 | 1.87 | 19 | 3 |
| 100 | 0.02342 | 2.34 | 15 | 2 |
| 120 | 0.02811 | 2.81 | 12 | 2 |

---

## Image Footprint

### Formulas

The footprint is the ground area captured by a single image:

```
footprint_w = GSD * W_px = (S_w * H) / f     (meters, along image width)
footprint_h = GSD * H_px = (S_h * H) / f     (meters, along image height)
footprint_area = footprint_w * footprint_h    (square meters)
```

### Reference Table (Mavic 2 Pro)

| Altitude (m) | Footprint W (m) | Footprint H (m) | Area (m^2) | Area (ha) |
|---------------|-----------------|-----------------|------------|-----------|
| 20 | 25.6 | 17.1 | 437 | 0.044 |
| 40 | 51.3 | 34.2 | 1,752 | 0.175 |
| 60 | 76.9 | 51.3 | 3,943 | 0.394 |
| 80 | 102.5 | 68.3 | 7,004 | 0.700 |
| 100 | 128.2 | 85.4 | 10,944 | 1.094 |

### Field of View (FOV)

```
FOV_w = 2 * arctan(S_w / (2 * f))    (horizontal FOV, radians)
FOV_h = 2 * arctan(S_h / (2 * f))    (vertical FOV, radians)
FOV_diagonal = 2 * arctan(sqrt(S_w^2 + S_h^2) / (2 * f))
```

**Mavic 2 Pro:**
```
FOV_w = 2 * arctan(13.2 / (2 * 10.3)) = 2 * arctan(0.6408) = 2 * 32.65 deg = 65.3 deg
FOV_h = 2 * arctan(8.8 / (2 * 10.3)) = 2 * arctan(0.4272) = 2 * 23.15 deg = 46.3 deg
```

---

## Flight Planning and Overlap

### Forward (Front/Longitudinal) Overlap

Front overlap is the percentage of image height shared between consecutive images along a flight line:

```
front_overlap = 1 - (trigger_distance / footprint_h)
trigger_distance = footprint_h * (1 - front_overlap)
```

**At 40 m, 70% overlap:**
```
trigger_distance = 34.2 * (1 - 0.70) = 10.25 m
```

### Side (Lateral) Overlap

Side overlap is the percentage of image width shared between adjacent flight lines:

```
side_overlap = 1 - (strip_spacing / footprint_w)
strip_spacing = footprint_w * (1 - side_overlap)
```

**At 40 m, 50% overlap:**
```
strip_spacing = 51.3 * (1 - 0.50) = 25.64 m
```

### Number of Images for Survey Area

```
# Flight lines (strips)
n_strips = ceil(area_cross_track / strip_spacing) + 1

# Images per strip
n_images_per_strip = ceil(area_along_track / trigger_distance) + 1

# Total images
total_images = n_strips * n_images_per_strip

# Total flight distance (excluding turns)
flight_distance = n_strips * area_along_track + (n_strips - 1) * strip_spacing

# Flight time (approximate, excluding turns)
flight_time = flight_distance / ground_speed + n_strips * turn_time
```

### Overlap and Structure from Motion

Minimum overlaps for reliable SfM reconstruction:
- Front overlap >= 60% (70-80% recommended)
- Side overlap >= 30% (40-60% recommended)

Each ground point appears in approximately:
```
images_per_point = (1 / (1 - front_overlap)) * (1 / (1 - side_overlap))
```

At 70/50: ~6.67 images per point (sufficient for SfM).
At 80/60: ~12.5 images per point (high redundancy, better 3D reconstruction).

---

## Motion Blur

### Model

During exposure, the drone moves across the ground. The resulting image blur (in pixels):

```
blur_px = ground_speed * exposure_time / GSD
```

Where:
- `ground_speed` = speed over ground (m/s), not airspeed
- `exposure_time` = shutter speed (seconds)
- `GSD` = ground sampling distance (m/px)

### Acceptable Blur Thresholds

| Blur (px) | Impact |
|-----------|--------|
| < 0.5 | Negligible -- no visible effect |
| 0.5 - 1.0 | Minor -- slight softening, acceptable for most detection |
| 1.0 - 2.0 | Moderate -- edge degradation, small objects affected |
| > 2.0 | Severe -- significant detail loss, detection accuracy impacted |

### Maximum Safe Exposure

For a target blur of 1.0 pixel:
```
max_exposure = GSD / ground_speed
```

For a target blur of 0.5 pixel:
```
max_exposure = 0.5 * GSD / ground_speed
```

### Wind Consideration

Effective ground speed includes wind:
```
ground_speed_effective = sqrt((airspeed + headwind)^2 + crosswind^2)
```

In the worst case (tailwind):
```
ground_speed_max = airspeed + wind_speed
```

Use the worst-case ground speed for blur calculations to ensure sufficient shutter speed in all flight directions.

---

## Orthorectification Pipeline

### Step-by-Step Process

#### Step 1: Image Collection with Metadata

Each image includes:
- GPS coordinates (latitude, longitude, altitude)
- Camera orientation (roll, pitch, yaw) from IMU
- Timestamp
- Camera settings (exposure, ISO, aperture)

#### Step 2: Feature Detection

Algorithms: SIFT (Scale-Invariant Feature Transform), ORB (Oriented FAST and Rotated BRIEF), SuperPoint (learned features).

Each image yields thousands of keypoints with descriptors.

#### Step 3: Feature Matching

Match keypoints between overlapping image pairs:
- Brute-force or FLANN-based matching
- Ratio test (Lowe's ratio) for outlier rejection
- RANSAC for geometric verification (fundamental/essential matrix)

#### Step 4: Bundle Adjustment

Simultaneously optimize:
- Camera positions (6 DoF per image: x, y, z, roll, pitch, yaw)
- 3D point positions (sparse reconstruction)
- Camera intrinsics (focal length, distortion)

Minimizes total reprojection error across all images and matched points.

#### Step 5: Dense Point Cloud

Multi-view stereo (MVS) algorithms:
- For each pixel in each image, find correspondences in overlapping images
- Triangulate to produce dense 3D points
- Typical density: 10-100 points per m^2

#### Step 6: Digital Surface Model (DSM)

- Interpolate dense point cloud to regular grid
- Grid resolution typically matches GSD
- Records elevation at each grid cell
- DSM includes vegetation and structures (vs DEM which is bare earth)

#### Step 7: Orthomosaic Generation

- Reproject each image to the ground plane using the DSM
- This removes perspective distortion: objects directly below the camera appear at correct scale, objects at the edges are corrected
- Blend overlapping ortho-images using seam line optimization and color balancing
- Output: single georeferenced image (GeoTIFF) with uniform GSD

### Software Comparison

| Software | Type | Strengths | Weaknesses | Cost |
|----------|------|-----------|------------|------|
| Pix4D | Desktop + cloud | High quality, good seam lines, robust BA | Expensive, requires powerful hardware for desktop | $350/month |
| DroneDeploy | Cloud-only | Fast, user-friendly, flight planning integration | Lower orthomosaic quality, seam line artifacts | $499/month |
| Agisoft Metashape | Desktop | High quality, flexible, academic pricing | Steep learning curve, slow on large datasets | $179 one-time |
| OpenDroneMap | Open-source desktop | Free, command-line friendly, Docker support | Variable quality, limited support | Free |
| WebODM | Open-source web UI | Free, accessible UI for ODM | Same limitations as ODM | Free |

### Thesis Finding

Pix4D orthomosaics led to ~0.07 higher F1 score than DroneDeploy orthomosaics for iguana detection. The quality difference is attributed to:
1. Better seam line placement (avoids cutting through iguanas)
2. More consistent color balancing
3. Better handling of moving objects (iguanas shifting between overlapping images)

---

## Coordinate Systems

### Image Coordinates (Pixels)

```
Origin: Top-left corner of image
u: Column index (0 to W_px-1), increasing rightward
v: Row index (0 to H_px-1), increasing downward
```

### Ground Coordinates (Meters, Local)

```
Origin: Center of orthomosaic (or arbitrary reference point)
X: Easting (meters)
Y: Northing (meters)
```

### Pixel-to-Ground Conversion

For orthomosaics with known GSD and origin:
```
X = origin_X + u * GSD
Y = origin_Y - v * GSD    (Y decreases as v increases)
```

### Geographic Coordinates (Lat/Lon)

GeoTIFF orthomosaics include affine transform metadata:
```python
import rasterio

with rasterio.open("orthomosaic.tif") as src:
    transform = src.transform
    # Pixel (col, row) to geographic (lon, lat):
    lon, lat = transform * (col, row)
```

---

## Tile Extraction from Orthomosaics

### Why Tile

Orthomosaics are too large for neural networks to process directly:
- Typical orthomosaic: 20,000-50,000 pixels per side
- Typical model input: 640x640 (YOLO) or 512x512 (HerdNet)
- Memory: a 30,000x30,000 RGB image requires ~2.7 GB in float32

### Tiling Strategy

```
tile_size_px = 640    # Match model input size
overlap_px = 128      # Overlap between tiles (20% of tile_size)
stride = tile_size_px - overlap_px  # 512

For each tile position (x, y):
  x = 0, stride, 2*stride, ...
  y = 0, stride, 2*stride, ...
  tile = image[y:y+tile_size_px, x:x+tile_size_px]
```

### Edge Handling

Tiles at image edges may be smaller than `tile_size_px`. Options:
1. **Pad:** Zero-pad to full tile size (may introduce edge artifacts)
2. **Skip:** Discard partial tiles (loses coverage at edges)
3. **Shift:** Adjust tile position to fill from edge inward (recommended)

### Detection Coordinate Mapping

After running detection on each tile, map tile-local coordinates to global (full-image) coordinates:

```
global_x = tile_origin_x + local_x
global_y = tile_origin_y + local_y
```

### Cross-Tile NMS

Detections near tile boundaries may be duplicated in adjacent tiles:
```
1. Collect all detections from all tiles (in global coordinates)
2. Apply NMS with standard IoU threshold
3. For point-based detection: merge points within a distance threshold
```
