---
name: geospatial-patterns
description: Load this skill when working with rasterio, GDAL, geopandas, orthomosaics, DEMs, GeoJSON, Shapefiles, CRS transformations, or any geospatial coordinate handling in HILDA. Also activated for Metashape integration and correspondence tracking.
---

# Geospatial Patterns for HILDA

## CRS — the most common source of bugs

**Always** check CRS before any spatial operation. The pipeline mixes:
- Drone imagery: typically WGS84 (EPSG:4326) from GPS
- Orthomosaics: often projected CRS (UTM zone for the Galapagos: EPSG:32715)
- DEM: must match orthomosaic CRS for DEM-based projection

```python
import rasterio
from rasterio.crs import CRS
import geopandas as gpd

# Always check CRS
with rasterio.open(ortho_path) as src:
    assert src.crs is not None, f"No CRS in {ortho_path}"
    crs = src.crs

# Reproject GeoDataFrame to match raster
detections_gdf = detections_gdf.to_crs(crs)

# Convert pixel coords → geographic coords
from rasterio.transform import xy
lon, lat = xy(src.transform, row=py, col=px)
```

## Metashape coordinate convention — NON-OBVIOUS GOTCHA

In Metashape's Python API, the camera center is the **translation vector directly**:
```python
# CORRECT: Metashape camera-to-chunk convention
camera_center = camera.transform.translation()  # This IS the 3D position

# WRONG: Standard computer vision convention
# camera_center = -R.T @ t   ← DO NOT USE with Metashape output
```

This is the single most common error in the correspondence tracking pipeline.

## Orthomosaic tiling

```python
from rasterio.windows import Window

def tile_orthomosaic(src_path: Path, tile_size: int = 1024, overlap: int = 64):
    """Yield (window, transform) pairs for tiled reading."""
    with rasterio.open(src_path) as src:
        for row in range(0, src.height, tile_size - overlap):
            for col in range(0, src.width, tile_size - overlap):
                window = Window(col, row, 
                               min(tile_size, src.width - col),
                               min(tile_size, src.height - row))
                transform = src.window_transform(window)
                yield src.read(window=window), transform
```

## GSD (Ground Sample Distance) computation

```python
def compute_gsd(altitude_m: float, focal_length_mm: float, 
                sensor_width_mm: float, image_width_px: int) -> float:
    """Returns GSD in cm/pixel."""
    return (altitude_m * sensor_width_mm) / (focal_length_mm * image_width_px) * 100
```

Typical Mavic 2 Pro values: focal=10.3mm, sensor_w=13.2mm, image_w=5472px

## Detection output → GeoJSON

```python
import json
from shapely.geometry import Point, mapping

def detections_to_geojson(detections: list[dict], crs: str = "EPSG:4326") -> dict:
    """Convert point detections to GeoJSON FeatureCollection."""
    features = []
    for det in detections:
        features.append({
            "type": "Feature",
            "geometry": mapping(Point(det['lon'], det['lat'])),
            "properties": {
                "confidence": det['conf'],
                "label": det['label'],
                "image_id": det.get('image_id'),
            }
        })
    return {"type": "FeatureCollection", "crs": {"type": "name", 
            "properties": {"name": crs}}, "features": features}
```

## KML export for DJI Pilot 2 (mission reproduction)

```python
import simplekml

def detections_to_kml(detections: list[dict], output_path: Path):
    kml = simplekml.Kml()
    for det in detections:
        pnt = kml.newpoint(name=f"iguana_{det['id']}", 
                          coords=[(det['lon'], det['lat'])])
        pnt.description = f"conf: {det['conf']:.2f}"
    kml.save(str(output_path))
```

## GDAL environment

Always install via conda:
```bash
conda install -c conda-forge gdal rasterio geopandas fiona pyproj
```
Never use pip for these — native bindings will fail.
