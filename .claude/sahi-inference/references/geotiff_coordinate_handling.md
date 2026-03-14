# GeoTIFF Coordinate Handling for SAHI Detections

## Why Coordinate Conversion Matters

SAHI produces detections in pixel coordinates (x, y in the image). For ecological
analysis, these must be converted to geographic coordinates (latitude, longitude)
using the geospatial metadata embedded in GeoTIFF files (drone orthomosaics,
satellite imagery).

---

## Reading GeoTIFF Metadata

```python
import rasterio

with rasterio.open('orthomosaic.tif') as src:
    # Coordinate Reference System
    print(f"CRS: {src.crs}")              # e.g., EPSG:4326 (WGS84) or EPSG:32617 (UTM 17N)

    # Affine transform: maps pixel (col, row) to geographic (x, y)
    print(f"Transform: {src.transform}")
    # Affine(a, b, c,
    #        d, e, f)
    # x_geo = a * col + b * row + c
    # y_geo = d * col + e * row + f

    # Image dimensions
    print(f"Size: {src.width} × {src.height}")

    # Ground Sample Distance (GSD)
    print(f"GSD X: {abs(src.transform.a):.4f} {src.crs.linear_units}")
    print(f"GSD Y: {abs(src.transform.e):.4f} {src.crs.linear_units}")

    # Bounding box
    print(f"Bounds: {src.bounds}")
```

---

## Pixel to Geographic Coordinate Conversion

### Single Point

```python
import rasterio

def pixel_to_geo(image_path, pixel_x, pixel_y):
    """Convert pixel coordinates to geographic coordinates."""
    with rasterio.open(image_path) as src:
        # rasterio.transform.xy expects (row, col) order
        x_geo, y_geo = rasterio.transform.xy(src.transform, pixel_y, pixel_x)
        return x_geo, y_geo
```

### SAHI Detection Centroids

```python
def detections_to_geo(result, image_path):
    """Convert SAHI detection centroids to geographic coordinates."""
    geo_detections = []

    with rasterio.open(image_path) as src:
        transform = src.transform
        crs = src.crs

        for pred in result.object_prediction_list:
            x1, y1, x2, y2 = pred.bbox.to_xyxy()

            # Centroid in pixel space
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Convert to geographic coordinates
            lon, lat = rasterio.transform.xy(transform, cy, cx)

            geo_detections.append({
                'lat': lat,
                'lon': lon,
                'confidence': pred.score.value,
                'category': pred.category.name,
                'bbox_px': [x1, y1, x2, y2],
            })

    return geo_detections, str(crs)
```

---

## Exporting Detections as GeoJSON

GeoJSON is the standard format for geographic vector data. It can be opened in
QGIS, Google Earth, and web mapping libraries (Leaflet, Mapbox).

```python
import json

def export_geojson(geo_detections, crs, output_path):
    """Export detections as GeoJSON FeatureCollection."""
    features = []
    for det in geo_detections:
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [det['lon'], det['lat']],  # GeoJSON uses [lon, lat]
            },
            'properties': {
                'confidence': det['confidence'],
                'category': det['category'],
                'bbox_px': det['bbox_px'],
            },
        }
        features.append(feature)

    geojson = {
        'type': 'FeatureCollection',
        'crs': {
            'type': 'name',
            'properties': {'name': crs},
        },
        'features': features,
    }

    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    print(f"Exported {len(features)} detections to {output_path}")
```

---

## Exporting as Shapefile (for GIS workflows)

```python
import geopandas as gpd
from shapely.geometry import Point

def export_shapefile(geo_detections, crs, output_path):
    """Export detections as a shapefile."""
    geometries = [Point(d['lon'], d['lat']) for d in geo_detections]
    gdf = gpd.GeoDataFrame(
        geo_detections,
        geometry=geometries,
        crs=crs,
    )
    gdf.to_file(output_path)
    print(f"Exported {len(gdf)} detections to {output_path}")
```

---

## Map Visualization with Folium

```python
import folium

def visualize_on_map(geo_detections, output_html='detection_map.html'):
    """Create an interactive map of detections."""
    if not geo_detections:
        print("No detections to visualize.")
        return

    # Center map on mean detection location
    mean_lat = sum(d['lat'] for d in geo_detections) / len(geo_detections)
    mean_lon = sum(d['lon'] for d in geo_detections) / len(geo_detections)

    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=18)

    for det in geo_detections:
        color = 'green' if det['confidence'] > 0.5 else 'orange' if det['confidence'] > 0.3 else 'red'
        folium.CircleMarker(
            location=[det['lat'], det['lon']],
            radius=4,
            color=color,
            fill=True,
            popup=f"{det['category']}: {det['confidence']:.2f}",
        ).add_to(m)

    m.save(output_html)
    print(f"Map saved to {output_html}")
```

---

## CRS Considerations

### Common CRS for Drone Surveys

| CRS | EPSG | Units | Use |
|-----|------|-------|-----|
| WGS 84 | 4326 | degrees | GPS coordinates, web maps |
| UTM zones | 326xx | meters | Local-scale measurements |
| Web Mercator | 3857 | meters | Google Maps, OpenStreetMap |

### Galápagos Islands
- **UTM Zone 15S** (EPSG:32715) — western islands (Fernandina, Isabela west)
- **UTM Zone 16S** (EPSG:32716) — central/eastern islands (Santa Cruz, Floreana)
- Orthomosaics from DJI drones are typically in WGS 84 (EPSG:4326)

### Converting Between CRS

```python
from pyproj import Transformer

# WGS 84 to UTM Zone 16S (for Galápagos central islands)
transformer = Transformer.from_crs('EPSG:4326', 'EPSG:32716', always_xy=True)
utm_x, utm_y = transformer.transform(lon, lat)
```

---

## Complete Pipeline: SAHI → GeoJSON

```python
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def detect_and_geolocate(image_path, model_path, output_dir, **sahi_kwargs):
    """Full pipeline: SAHI detection → georeferenced output."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect
    model = AutoDetectionModel.from_pretrained(
        model_type='yolov8', model_path=model_path,
        confidence_threshold=0.25, device='cuda:0',
    )
    result = get_sliced_prediction(image=image_path, detection_model=model, **sahi_kwargs)

    # Convert to geographic coordinates
    geo_dets, crs = detections_to_geo(result, image_path)

    # Export
    export_geojson(geo_dets, crs, output_dir / 'detections.geojson')
    visualize_on_map(geo_dets, str(output_dir / 'map.html'))

    print(f"Detected {len(geo_dets)} animals in {Path(image_path).name}")
    return geo_dets
```
