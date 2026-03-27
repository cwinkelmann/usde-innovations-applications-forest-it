# Label Studio Integration for Active Learning

## Overview

Label Studio is a flexible open-source annotation tool that supports a wide range
of annotation types. It's particularly useful when the active learning workflow
involves mixed annotation types (bounding boxes + text labels + classification).

**Website:** https://labelstud.io
**Repository:** https://github.com/HumanSignal/label-studio
**License:** Apache 2.0

---

## Installation

### Quick Start (pip)

```bash
pip install label-studio
label-studio start
# Opens at http://localhost:8080
```

### Docker

```bash
docker run -p 8080:8080 \
    -v label-studio-data:/label-studio/data \
    heartexlabs/label-studio
```

### Python SDK

```bash
pip install label-studio-sdk
```

---

## SDK Usage

### Authentication

```python
from label_studio_sdk import Client

# Connect to Label Studio
ls = Client(url='http://localhost:8080', api_key='YOUR_API_KEY')

# Get API key from Label Studio UI: Account & Settings → Access Token
```

### Project Management

```python
# Create a project for bounding box detection
project = ls.start_project(
    title='Marine Iguana Detection - AL Round 1',
    label_config='''
    <View>
      <Image name="image" value="$image"/>
      <RectangleLabels name="label" toName="image">
        <Label value="iguana"/>
        <Label value="ignore"/>
      </RectangleLabels>
    </View>
    '''
)

# For point annotation (HerdNet-compatible)
project_points = ls.start_project(
    title='Iguana Point Counting',
    label_config='''
    <View>
      <Image name="image" value="$image"/>
      <KeyPointLabels name="label" toName="image">
        <Label value="iguana"/>
      </KeyPointLabels>
    </View>
    '''
)

# List projects
projects = ls.list_projects()
for p in projects:
    print(f"  {p.id}: {p.title}")
```

### Uploading Images

```python
# Method 1: Local file references (requires local file serving)
tasks = []
for img_path in image_paths:
    tasks.append({
        'data': {'image': f'/data/local-files/?d={img_path}'}
    })
project.import_tasks(tasks)

# Method 2: URL references (images served via HTTP)
tasks = []
for img_url in image_urls:
    tasks.append({
        'data': {'image': img_url}
    })
project.import_tasks(tasks)
```

### Uploading Pre-Annotations

Label Studio uses a specific prediction format with percentage-based coordinates.

```python
def upload_predictions(project, predictions, image_width, image_height):
    """Upload model predictions as pre-annotations.

    Args:
        project: Label Studio project object
        predictions: list of dicts with 'image_path' and 'detections'
            each detection has 'bbox' [x1,y1,x2,y2] in pixels, 'class', 'score'
        image_width: width of images in pixels
        image_height: height of images in pixels
    """
    tasks = []
    for item in predictions:
        results = []
        for det in item['detections']:
            x1, y1, x2, y2 = det['bbox']

            # Convert pixel coords to percentage (Label Studio format)
            results.append({
                'type': 'rectanglelabels',
                'from_name': 'label',
                'to_name': 'image',
                'original_width': image_width,
                'original_height': image_height,
                'value': {
                    'x': (x1 / image_width) * 100,
                    'y': (y1 / image_height) * 100,
                    'width': ((x2 - x1) / image_width) * 100,
                    'height': ((y2 - y1) / image_height) * 100,
                    'rectanglelabels': [det['class']],
                },
                'score': det['score'],
            })

        tasks.append({
            'data': {'image': item['image_path']},
            'predictions': [{'result': results}],
        })

    project.import_tasks(tasks)
    print(f"Uploaded {len(tasks)} tasks with pre-annotations")
```

### Downloading Annotations

```python
def download_annotations(project, export_type='COCO'):
    """Download completed annotations from Label Studio.

    Args:
        project: Label Studio project object
        export_type: 'COCO', 'YOLO', 'VOC', 'JSON', 'CSV'

    Returns:
        Exported annotations in the requested format
    """
    export = project.export_tasks(export_type=export_type)
    return export


def download_coco_annotations(project, output_path):
    """Download annotations in COCO format and save to file."""
    coco = download_annotations(project, export_type='COCO')

    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2)

    n_images = len(coco.get('images', []))
    n_annotations = len(coco.get('annotations', []))
    print(f"Exported {n_annotations} annotations from {n_images} images to {output_path}")

    return coco
```

---

## Active Learning Workflow with Label Studio

### Complete Round Function

```python
from pathlib import Path
import json
from label_studio_sdk import Client


def run_ls_annotation_round(
    ls_url: str,
    api_key: str,
    round_num: int,
    image_paths: list,
    predictions: dict,
    labels: list,
    image_width: int,
    image_height: int,
    output_dir: str,
):
    """Execute one annotation round using Label Studio.

    Args:
        ls_url: Label Studio URL
        api_key: API key
        round_num: Active learning round number
        image_paths: List of image paths to annotate
        predictions: Dict mapping image filename to detection list
        labels: List of label names
        image_width: Image width in pixels
        image_height: Image height in pixels
        output_dir: Where to save exported annotations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ls = Client(url=ls_url, api_key=api_key)

    # 1. Create project
    label_options = '\n'.join(f'        <Label value="{l}"/>' for l in labels)
    project = ls.start_project(
        title=f'AL Round {round_num}',
        label_config=f'''
        <View>
          <Image name="image" value="$image"/>
          <RectangleLabels name="label" toName="image">
    {label_options}
          </RectangleLabels>
        </View>
        '''
    )
    print(f"Created project: AL Round {round_num} (ID: {project.id})")

    # 2. Upload images with pre-annotations
    pred_items = []
    for img_path in image_paths:
        img_name = Path(img_path).name
        dets = predictions.get(img_name, [])
        pred_items.append({
            'image_path': str(img_path),
            'detections': dets,
        })

    upload_predictions(project, pred_items, image_width, image_height)

    # 3. Wait for annotation
    print(f"\nAnnotation project ready at: {ls_url}/projects/{project.id}")
    print("Please review and correct annotations in Label Studio.")
    input("Press Enter when annotation is complete...")

    # 4. Export corrected annotations
    export_path = output_dir / f'round_{round_num}_annotations.json'
    coco = download_coco_annotations(project, export_path)

    return coco
```

---

## Label Studio vs. CVAT for Active Learning

### Choose Label Studio when:
- You need **mixed annotation types** (boxes + text + classification)
- **Quick single-user setup** is a priority (`pip install` vs Docker)
- You want **flexible labeling interfaces** (custom layouts)
- Your team uses **Label Studio Enterprise** for project management

### Choose CVAT when:
- You only need **geometric annotations** (boxes, polygons, points)
- **Multi-annotator workflows** with review/QA are important
- **Video annotation** with frame tracking is needed
- You want **tighter SDK integration** for bulk operations

### For this course:
CVAT is recommended as the default because:
1. Detection (boxes/points) is the primary annotation type
2. The SDK supports bulk pre-annotation import well
3. Docker deployment is straightforward for lab environments
4. Multi-student annotation projects are well-supported
