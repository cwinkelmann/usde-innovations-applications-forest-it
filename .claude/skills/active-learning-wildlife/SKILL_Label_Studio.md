---
name: label-studio-hilda
description: >
  Complete reference for the HILDA Label Studio pipeline: Docker setup,
  programmatic project creation, direct image upload, pre-annotation injection
  (bounding boxes + keypoints + classification), and annotation export/parsing
  back to training format. Use this skill whenever the user asks about Label
  Studio setup, HILDA annotation workflows, pushing MegaDetector/HerdNet
  predictions into Label Studio, or retrieving corrected annotations.
---

# Label Studio — HILDA Pipeline Skill

## 1. Architecture Overview

```
Drone tiles (PNG/JPG)
       │
       ▼
HerdNet / YOLO inference  →  detections (pixel coords)
       │
       ▼
[this pipeline]
       │
  ┌────┴──────────────────────────────────────────────┐
  │  upload_images_to_project()  ← pushes image bytes │
  │  attach_predictions()        ← pre-annotates tasks │
  └────┬──────────────────────────────────────────────┘
       │
       ▼
Label Studio (Docker + Postgres)
  └── biologist corrects boxes / points in browser
       │
       ▼
export_annotations()  →  list[dict]  →  training-ready COCO / CSV
```

Key principle: **images are pushed (uploaded) directly to Label Studio** — no
separate file server, no S3, no local-mount required. This keeps the workflow
self-contained and portable.

---

## 2. Docker Setup

### 2a. Quick local dev (SQLite, no persistence between restarts)

```bash
docker run -it -p 8080:8080 \
  -v "$(pwd)/ls_data:/label-studio/data" \
  heartexlabs/label-studio:latest
```

### 2b. Production server (Postgres, persistent, recommended for HILDA)

Create `docker-compose.yml`:

```yaml
version: "3.8"

services:
  db:
    image: postgres:15
    restart: unless-stopped
    environment:
      POSTGRES_DB:       labelstudio
      POSTGRES_USER:     ls_user
      POSTGRES_PASSWORD: ${LS_DB_PASS:-changeme}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ls_user -d labelstudio"]
      interval: 5s
      timeout: 5s
      retries: 10

  label-studio:
    image: heartexlabs/label-studio:latest
    restart: unless-stopped
    ports:
      - "8080:8080"
    depends_on:
      db:
        condition: service_healthy
    environment:
      # Postgres
      DJANGO_DB:        default
      POSTGRE_NAME:     labelstudio
      POSTGRE_USER:     ls_user
      POSTGRE_PASSWORD: ${LS_DB_PASS:-changeme}
      POSTGRE_HOST:     db
      POSTGRE_PORT:     5432
      # Instance URL (important for correct file URLs in responses)
      LABEL_STUDIO_HOST: ${LABEL_STUDIO_HOST:-http://localhost:8080}
    volumes:
      - ls_data:/label-studio/data

volumes:
  postgres_data:
  ls_data:
```

Start/stop:
```bash
docker compose up -d          # start detached
docker compose logs -f        # follow logs
docker compose down           # stop (data persists in volumes)
docker compose down --volumes # !! wipes all data !!
```

First-run admin account: visit http://localhost:8080 and register with any
email + password. **Copy your API token** from
Account & Settings → Access Token before running any scripts.

### 2c. Environment file (.env, never commit to git)

```bash
# .env
LABEL_STUDIO_HOST=http://your-server-ip:8080
LABEL_STUDIO_API_KEY=<token-from-account-settings>
LS_DB_PASS=a-secure-password
```

Load in Python:
```python
from dotenv import load_dotenv
import os
load_dotenv()
LS_URL = os.environ["LABEL_STUDIO_HOST"]
API_KEY = os.environ["LABEL_STUDIO_API_KEY"]
```

---

## 3. SDK Installation and Client Setup

```bash
pip install "label-studio-sdk>=2.0.0" python-dotenv requests Pillow
```

**CRITICAL**: SDK v2 (released Aug 2025) has breaking changes from SDK v1.
Always use the v2 API shown in this skill. The old `Client` / `get_project()`
pattern is deprecated.

```python
from label_studio_sdk import LabelStudio

client = LabelStudio(base_url=LS_URL, api_key=API_KEY)
# Test connection:
projects = list(client.projects.list())
print(f"Connected — {len(projects)} existing projects")
```

---

## 4. Label Config XML Templates

Label Studio uses XML to define the annotation interface.
Key rules:
- Every `<Control>` tag has `name` and `toName` attributes
- `name` is the key that appears in exported results
- `toName` must match the `name` of the `<Object>` tag (e.g. `<Image>`)
- `value="$image"` binds to the `"image"` key in task `data` dict

### 4a. Bounding boxes + classification (for MegaDetector / YOLO style)

```xml
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"
         rotateControl="false" brightnessControl="true"/>

  <RectangleLabels name="bbox" toName="image" showInline="true">
    <Label value="iguana"          background="#E74C3C"/>
    <Label value="iguana_juvenile" background="#F39C12"/>
    <Label value="iguana_uncertain" background="#95A5A6"/>
    <Label value="other"           background="#3498DB"/>
  </RectangleLabels>
</View>
```

### 4b. Point annotations (for HerdNet FIDT-style point supervision)

```xml
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>

  <KeyPointLabels name="kp" toName="image">
    <Label value="iguana"           background="#E74C3C"/>
    <Label value="iguana_uncertain" background="#F39C12"/>
  </KeyPointLabels>
</View>
```

### 4c. Combined: points + bounding boxes (HILDA default)

```xml
<View>
  <Style>
    .lsf-main-content.lsf-requesting .prompt::before { content: ' '; }
  </Style>

  <Image name="image" value="$image"
         zoom="true" zoomControl="true"
         brightnessControl="true" contrastControl="true"
         crossOrigin="anonymous"/>

  <Header value="Bounding Boxes"/>
  <RectangleLabels name="bbox" toName="image" showInline="true">
    <Label value="iguana"           background="#E74C3C"/>
    <Label value="iguana_juvenile"  background="#E67E22"/>
    <Label value="iguana_uncertain" background="#95A5A6"/>
  </RectangleLabels>

  <Header value="Point Annotations (for count verification)"/>
  <KeyPointLabels name="kp" toName="image">
    <Label value="iguana"           background="#E74C3C"/>
    <Label value="iguana_uncertain" background="#F39C12"/>
  </KeyPointLabels>
</View>
```

### 4d. Bounding boxes + choice classification (species + confidence rating)

```xml
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>

  <RectangleLabels name="bbox" toName="image">
    <Label value="iguana"           background="#E74C3C"/>
    <Label value="iguana_juvenile"  background="#E67E22"/>
    <Label value="other_animal"     background="#3498DB"/>
  </RectangleLabels>

  <Choices name="image_quality" toName="image" choice="single"
           showInline="true">
    <Choice value="good"/>
    <Choice value="blurry"/>
    <Choice value="partial"/>
    <Choice value="unusable"/>
  </Choices>
</View>
```

---

## 5. Project Creation

```python
def create_hilda_project(client, title, label_config, description=""):
    """Create a new Label Studio project and return it."""
    project = client.projects.create(
        title=title,
        label_config=label_config,
        description=description,
        # Show predictions to annotators automatically:
        show_annotation_history=True,
        # Copy prediction to annotation on open (key for review workflow):
        # (set in project settings UI: ML > "Show predictions to annotators")
    )
    print(f"Created project '{project.title}' — id={project.id}")
    return project


def get_or_create_project(client, title, label_config):
    """Idempotent: return existing project by title or create new."""
    for p in client.projects.list():
        if p.title == title:
            print(f"Found existing project '{title}' — id={p.id}")
            return p
    return create_hilda_project(client, title, label_config)
```

---

## 6. Image Upload and Task Creation

Label Studio stores uploaded images in its internal data volume.
Images are uploaded via multipart POST to `/api/projects/{id}/import`.
This creates tasks AND stores the files atomically.

### 6a. Upload a single image file and get back task ID

```python
import requests
from pathlib import Path

def upload_image_as_task(ls_url: str, api_key: str, project_id: int,
                          image_path: Path) -> int:
    """
    Upload a single image file to a Label Studio project.
    Returns the created task ID.
    """
    url = f"{ls_url}/api/projects/{project_id}/import"
    headers = {"Authorization": f"Token {api_key}"}

    with open(image_path, "rb") as f:
        resp = requests.post(
            url,
            headers=headers,
            files={"file": (image_path.name, f, "image/jpeg")},
        )
    resp.raise_for_status()
    result = resp.json()
    # Returns: {"task_count": N, "annotation_count": 0, "prediction_count": 0, "duration": ..., "task_ids": [...]}
    task_ids = result.get("task_ids", [])
    if not task_ids:
        raise RuntimeError(f"Upload returned no task IDs: {result}")
    return task_ids[0]
```

### 6b. Batch upload with a progress bar

```python
from tqdm import tqdm
import time

def upload_images_batch(ls_url: str, api_key: str, project_id: int,
                         image_paths: list[Path],
                         sleep_between: float = 0.05) -> list[int]:
    """
    Upload a list of images to LS project.
    Returns list of task IDs in the same order as image_paths.
    """
    task_ids = []
    url = f"{ls_url}/api/projects/{project_id}/import"
    headers = {"Authorization": f"Token {api_key}"}

    for path in tqdm(image_paths, desc="Uploading images"):
        with open(path, "rb") as f:
            resp = requests.post(
                url,
                headers=headers,
                files={"file": (path.name, f, "image/jpeg")},
            )
        resp.raise_for_status()
        ids = resp.json().get("task_ids", [])
        task_ids.extend(ids)
        time.sleep(sleep_between)   # be gentle with the server

    return task_ids
```

### 6c. Alternative: JSON import with image URLs (when images are publicly accessible)

```python
def import_tasks_from_urls(client, project_id: int,
                            image_urls: list[str]) -> list[int]:
    """Import tasks from HTTP(S) URLs — no upload needed."""
    tasks = [{"data": {"image": url}} for url in image_urls]
    result = client.projects.import_tasks(id=project_id, request=tasks)
    return result  # returns import job info
```

---

## 7. Pre-annotation Format (Predictions)

### CRITICAL coordinate convention
Label Studio uses **percentages of image dimensions**, NOT pixel coordinates.

```python
def px_to_pct(x_px, y_px, w_px, h_px, img_w, img_h):
    """Convert pixel bbox to Label Studio percentage format."""
    return {
        "x":      x_px / img_w * 100,
        "y":      y_px / img_h * 100,
        "width":  w_px / img_w * 100,
        "height": h_px / img_h * 100,
    }

def pct_to_px(x_pct, y_pct, w_pct, h_pct, img_w, img_h):
    """Convert Label Studio percentage format back to pixel coordinates."""
    return {
        "x1": x_pct / 100 * img_w,
        "y1": y_pct / 100 * img_h,
        "x2": (x_pct + w_pct) / 100 * img_w,
        "y2": (y_pct + h_pct) / 100 * img_h,
    }
```

For keypoints (points), Label Studio uses x/y in % of image dimensions with
a `width` field that controls the dot radius (usually 0.5–2.0).

### 7a. Build a bbox prediction result

```python
import uuid

def make_bbox_result(x1_px, y1_px, x2_px, y2_px,
                     label: str, img_w: int, img_h: int,
                     score: float = 1.0) -> dict:
    """
    Single bounding box result dict for Label Studio predictions.
    Coords are pixel x1,y1 (top-left) and x2,y2 (bottom-right).
    """
    return {
        "id":        str(uuid.uuid4())[:8],
        "from_name": "bbox",       # must match XML <RectangleLabels name=...>
        "to_name":   "image",      # must match XML <Image name=...>
        "type":      "rectanglelabels",
        "value": {
            "x":               (x1_px / img_w) * 100,
            "y":               (y1_px / img_h) * 100,
            "width":           ((x2_px - x1_px) / img_w) * 100,
            "height":          ((y2_px - y1_px) / img_h) * 100,
            "rotation":        0,
            "rectanglelabels": [label],
        },
        "score": score,
    }
```

### 7b. Build a keypoint result

```python
def make_keypoint_result(x_px, y_px, label: str,
                          img_w: int, img_h: int,
                          score: float = 1.0,
                          dot_size: float = 1.0) -> dict:
    """Single keypoint result dict."""
    return {
        "id":        str(uuid.uuid4())[:8],
        "from_name": "kp",        # must match XML <KeyPointLabels name=...>
        "to_name":   "image",
        "type":      "keypointlabels",
        "value": {
            "x":               (x_px / img_w) * 100,
            "y":               (y_px / img_h) * 100,
            "width":           dot_size,
            "keypointlabels":  [label],
        },
        "score": score,
    }
```

### 7c. Assemble a full prediction object

```python
def make_prediction(results: list[dict], model_version: str,
                     score: float | None = None) -> dict:
    """
    Wrap result dicts into a LS prediction object.
    `score` is the overall image-level confidence (optional, used for sorting).
    """
    avg_score = (
        score if score is not None
        else sum(r.get("score", 1.0) for r in results) / max(len(results), 1)
    )
    return {
        "model_version": model_version,
        "score":         avg_score,
        "result":        results,
    }
```

---

## 8. Attaching Predictions to Existing Tasks

After uploading images (section 6), attach predictions via the REST API.
The SDK v2 `client.predictions.create()` call:

```python
def attach_prediction_to_task(client, task_id: int,
                               result: list[dict],
                               model_version: str,
                               score: float = 0.0):
    """Push a pre-annotation prediction to an already-created task."""
    client.predictions.create(
        task=task_id,
        result=result,
        model_version=model_version,
        score=score,
    )
```

### Full pipeline: upload + predict in one loop

```python
from PIL import Image as PILImage

def upload_and_preannotate(ls_url: str, api_key: str, client,
                            project_id: int,
                            items: list[dict],
                            model_version: str = "herdnet-v1"):
    """
    items: list of dicts with keys:
        path       : Path  — local image file
        detections : list of dicts with x1,y1,x2,y2,label,score (pixels)

    Uploads each image, then attaches predictions.
    Returns list of task IDs.
    """
    all_task_ids = []
    for item in tqdm(items, desc="Upload + pre-annotate"):
        path = item["path"]
        dets = item.get("detections", [])

        # Get image dimensions for coordinate conversion
        with PILImage.open(path) as img:
            img_w, img_h = img.size

        # 1. Upload image → get task id
        task_id = upload_image_as_task(ls_url, api_key, project_id, path)
        all_task_ids.append(task_id)

        # 2. Build results
        if not dets:
            continue   # no predictions for this image — leave blank

        results = []
        for det in dets:
            if "x2" in det:  # bbox
                results.append(make_bbox_result(
                    det["x1"], det["y1"], det["x2"], det["y2"],
                    det.get("label", "iguana"), img_w, img_h,
                    det.get("score", 1.0),
                ))
            else:  # point (x_center, y_center)
                results.append(make_keypoint_result(
                    det["x"], det["y"],
                    det.get("label", "iguana"), img_w, img_h,
                    det.get("score", 1.0),
                ))

        avg_score = sum(d.get("score", 1.0) for d in dets) / len(dets)

        # 3. Attach prediction
        attach_prediction_to_task(
            client, task_id,
            result=results,
            model_version=model_version,
            score=avg_score,
        )

    return all_task_ids
```

---

## 9. Enabling Pre-annotations in the UI

After pushing predictions, annotators must see them automatically.
Configure via the API (or UI: Settings → Machine Learning):

```python
def enable_predictions_for_annotators(client, project_id: int,
                                        model_version: str):
    """
    Tell LS to show predictions from model_version as pre-annotations.
    Annotators will see boxes pre-filled; they just correct/confirm.
    """
    client.projects.update(
        id=project_id,
        show_collab_predictions=True,      # show predictions to annotators
        model_version=model_version,       # which version to show
    )
```

---

## 10. Exporting Annotations

### 10a. Export all completed tasks as JSON

```python
def export_annotations_json(ls_url: str, api_key: str,
                              project_id: int) -> list[dict]:
    """
    Export all tasks with at least one annotation.
    Returns raw Label Studio JSON export format.
    """
    url = f"{ls_url}/api/projects/{project_id}/export"
    resp = requests.get(
        url,
        headers={"Authorization": f"Token {api_key}"},
        params={"exportType": "JSON"},
    )
    resp.raise_for_status()
    return resp.json()  # list of task dicts
```

### 10b. Export only labeled tasks via SDK

```python
def get_labeled_tasks(client, project_id: int) -> list:
    """Return tasks that have at least one human annotation."""
    return list(client.tasks.list(
        project=project_id,
        only_with_annotations=True,
    ))
```

---

## 11. Parsing Annotations Back to Training Format

Raw LS export JSON structure:
```json
[
  {
    "id": 42,
    "data": {"image": "/data/upload/1/my_tile.jpg"},
    "annotations": [
      {
        "id": 7,
        "completed_by": 1,
        "result": [
          {
            "type": "rectanglelabels",
            "from_name": "bbox",
            "to_name": "image",
            "original_width": 640,
            "original_height": 640,
            "value": {
              "x": 10.5, "y": 20.3, "width": 15.0, "height": 8.0,
              "rotation": 0,
              "rectanglelabels": ["iguana"]
            }
          }
        ]
      }
    ],
    "predictions": [ ... ]
  }
]
```

### 11a. Parse to flat DataFrame

```python
import pandas as pd

def parse_annotations_to_df(export_json: list[dict]) -> pd.DataFrame:
    """
    Convert LS JSON export to a flat DataFrame.
    Pixel coordinates are restored using original_width / original_height.

    Returns DataFrame with columns:
        task_id, image_url, annotator_id,
        label, type,
        x1_px, y1_px, x2_px, y2_px,    (bboxes; NaN for keypoints)
        x_px, y_px,                      (keypoints; NaN for bboxes)
        x_pct, y_pct, w_pct, h_pct      (raw LS percentages)
    """
    rows = []
    for task in export_json:
        task_id  = task["id"]
        img_url  = task["data"].get("image", "")

        for ann in task.get("annotations", []):
            annotator = ann.get("completed_by")
            for region in ann.get("result", []):
                rtype = region.get("type", "")
                v     = region.get("value", {})
                orig_w = region.get("original_width",  640)
                orig_h = region.get("original_height", 640)

                row = {
                    "task_id":      task_id,
                    "image_url":    img_url,
                    "annotator_id": annotator,
                    "type":         rtype,
                    "label":        None,
                    "x_pct":        v.get("x"),
                    "y_pct":        v.get("y"),
                    "w_pct":        v.get("width"),
                    "h_pct":        v.get("height"),
                    "x1_px":        None, "y1_px": None,
                    "x2_px":        None, "y2_px": None,
                    "x_px":         None, "y_px":  None,
                }

                if rtype == "rectanglelabels":
                    row["label"] = v.get("rectanglelabels", [None])[0]
                    row["x1_px"] = v["x"]      / 100 * orig_w
                    row["y1_px"] = v["y"]      / 100 * orig_h
                    row["x2_px"] = (v["x"] + v["width"])  / 100 * orig_w
                    row["y2_px"] = (v["y"] + v["height"]) / 100 * orig_h

                elif rtype == "keypointlabels":
                    row["label"] = v.get("keypointlabels", [None])[0]
                    row["x_px"]  = v["x"] / 100 * orig_w
                    row["y_px"]  = v["y"] / 100 * orig_h

                rows.append(row)

    return pd.DataFrame(rows)
```

### 11b. Parse to COCO-style format (for object detection training)

```python
def parse_to_coco(export_json: list[dict],
                   label_map: dict[str, int] | None = None) -> dict:
    """
    Convert LS export to minimal COCO JSON format.

    label_map: e.g. {"iguana": 1, "iguana_juvenile": 2}
               If None, labels are auto-assigned starting from 1.
    """
    if label_map is None:
        # Auto-discover labels
        all_labels = set()
        for task in export_json:
            for ann in task.get("annotations", []):
                for r in ann.get("result", []):
                    v = r.get("value", {})
                    for key in ("rectanglelabels", "keypointlabels"):
                        all_labels.update(v.get(key, []))
        label_map = {lbl: i + 1 for i, lbl in enumerate(sorted(all_labels))}

    categories = [{"id": v, "name": k} for k, v in label_map.items()]
    images_coco, annotations_coco = [], []
    ann_id = 1

    for task in export_json:
        img_id  = task["id"]
        img_url = task["data"].get("image", "")

        # Get image size from first annotation's original_width/height
        orig_w, orig_h = 640, 640
        for ann in task.get("annotations", []):
            for r in ann.get("result", []):
                orig_w = r.get("original_width",  orig_w)
                orig_h = r.get("original_height", orig_h)
                break
            break

        images_coco.append({
            "id":        img_id,
            "file_name": img_url.split("/")[-1],
            "width":     orig_w,
            "height":    orig_h,
        })

        for ann in task.get("annotations", []):
            for region in ann.get("result", []):
                if region.get("type") != "rectanglelabels":
                    continue
                v     = region["value"]
                ow    = region.get("original_width",  orig_w)
                oh    = region.get("original_height", orig_h)
                label = v.get("rectanglelabels", ["unknown"])[0]
                cat_id = label_map.get(label, 0)

                x1 = v["x"]     / 100 * ow
                y1 = v["y"]     / 100 * oh
                bw = v["width"] / 100 * ow
                bh = v["height"]/ 100 * oh

                annotations_coco.append({
                    "id":          ann_id,
                    "image_id":    img_id,
                    "category_id": cat_id,
                    "bbox":        [x1, y1, bw, bh],   # COCO: x,y,w,h
                    "area":        bw * bh,
                    "iscrowd":     0,
                })
                ann_id += 1

    return {
        "categories":   categories,
        "images":       images_coco,
        "annotations":  annotations_coco,
    }
```

### 11c. Parse to HerdNet point annotation format

```python
def parse_to_herdnet_points(export_json: list[dict]) -> pd.DataFrame:
    """
    Extract keypoint annotations into a DataFrame compatible with HerdNet
    training pipeline (x, y in pixel coordinates, label, image path).
    """
    rows = []
    for task in export_json:
        img_url = task["data"].get("image", "")
        for ann in task.get("annotations", []):
            for region in ann.get("result", []):
                if region.get("type") != "keypointlabels":
                    continue
                v     = region["value"]
                orig_w = region.get("original_width",  640)
                orig_h = region.get("original_height", 640)
                rows.append({
                    "image_path": img_url,
                    "x":          v["x"] / 100 * orig_w,
                    "y":          v["y"] / 100 * orig_h,
                    "label":      v.get("keypointlabels", ["iguana"])[0],
                    "task_id":    task["id"],
                })
    return pd.DataFrame(rows)
```

---

## 12. Common Gotchas

| Issue | Symptom | Fix |
|-------|---------|-----|
| Coordinate mismatch | Boxes appear at wrong position | All coords must be **% of image dims**, not pixels |
| `from_name` / `to_name` wrong | Predictions silently ignored | Must exactly match XML tag `name` attributes |
| SDK v1 vs v2 | `AttributeError: Client has no attribute projects` | Use `LabelStudio` not `Client`; pin `sdk>=2.0.0` |
| Predictions read-only | Can't edit predictions in UI | Correct — annotators work on *annotations* copied from predictions |
| Images not visible | Broken image in task | Check `LABEL_STUDIO_HOST` env var matches the URL your browser uses |
| Token auth fails | 401 on all API calls | Header must be `"Authorization": "Token <key>"` not `"Bearer <key>"` |
| Duplicate tasks | Same image imported twice | Check task count before uploading; use `get_or_create` pattern |
| SQLite bottleneck | Slow with >10k tasks or concurrent users | Switch to Postgres compose setup |
| `original_width` missing | Export coords wrong | LS fills this on annotation; if blank, store img dims in task `meta` |

### Storing image dimensions in task meta (best practice)

```python
# When creating tasks via JSON import, store dimensions in meta:
tasks = [
    {
        "data": {"image": url},
        "meta": {"width": img_w, "height": img_h, "filename": fname},
        "predictions": [prediction],
    }
    for url, img_w, img_h, fname, prediction in items
]
client.projects.import_tasks(id=project_id, request=tasks)

# Then in export parsing, fall back to meta if original_width is missing:
orig_w = region.get("original_width") or task.get("meta", {}).get("width", 640)
```

---

## 13. Useful REST Endpoints Reference

All require header: `Authorization: Token <api_key>`

```
GET    /api/projects/                          list all projects
POST   /api/projects/                          create project
GET    /api/projects/{id}/                     get project info
PATCH  /api/projects/{id}/                     update project settings

POST   /api/projects/{id}/import               upload image file OR import JSON tasks
GET    /api/projects/{id}/export?exportType=JSON  export annotations

GET    /api/tasks/?project={id}                list tasks
POST   /api/tasks/                             create task (JSON)
GET    /api/tasks/{id}/                        get single task with annotations
DELETE /api/tasks/{id}/                        delete task

POST   /api/predictions/                       create prediction for task
GET    /api/predictions/?task={id}             list predictions for task
DELETE /api/predictions/{id}/                  delete prediction

GET    /api/annotations/?task={id}             list annotations for task
GET    /api/annotations/{id}/                  get annotation detail
DELETE /api/annotations/{id}/                  delete annotation
```

---

## 14. Full HILDA Workflow in ~30 Lines

```python
import os
from pathlib import Path
from label_studio_sdk import LabelStudio
from dotenv import load_dotenv

load_dotenv()
client = LabelStudio(base_url=os.environ["LABEL_STUDIO_HOST"],
                     api_key=os.environ["LABEL_STUDIO_API_KEY"])

LABEL_CONFIG = """
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <RectangleLabels name="bbox" toName="image">
    <Label value="iguana" background="#E74C3C"/>
    <Label value="iguana_uncertain" background="#95A5A6"/>
  </RectangleLabels>
</View>
"""

# 1. Create project
project = get_or_create_project(client, "Floreana 2024", LABEL_CONFIG)

# 2. Prepare items (your detection pipeline produces these)
items = [
    {
        "path": Path("tiles/tile_0001.jpg"),
        "detections": [
            {"x1": 120, "y1": 80, "x2": 180, "y2": 140,
             "label": "iguana", "score": 0.91},
        ],
    },
    # ... more tiles ...
]

# 3. Upload + pre-annotate
task_ids = upload_and_preannotate(
    os.environ["LABEL_STUDIO_HOST"],
    os.environ["LABEL_STUDIO_API_KEY"],
    client, project.id, items,
    model_version="herdnet-dla169-v1",
)

# 4. (biologist corrects in browser)

# 5. Export + parse
raw = export_annotations_json(
    os.environ["LABEL_STUDIO_HOST"],
    os.environ["LABEL_STUDIO_API_KEY"],
    project.id,
)
df = parse_annotations_to_df(raw)
df.to_csv("corrected_annotations.csv", index=False)
```
