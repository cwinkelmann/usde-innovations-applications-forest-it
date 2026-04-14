"""MegaDetector (ultralytics YOLO) wrapper.

Matches week1/practicals/practical_02_megadetector_ultralytics.ipynb:
uses `ultralytics.YOLO` directly against an MD v1000 `.pt` checkpoint.
Ultralytics cannot auto-download MD v1000 from its own CDN — weights come
from the agentmorris/MegaDetector GitHub release, which we fetch on first use.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Callable, Iterable

LABELS = {0: "animal", 1: "person", 2: "vehicle"}

MD_V1000_URLS = {
    "md_v1000.0.0-larch.pt": "https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-larch.pt",
    "md_v1000.0.0-sorrel.pt": "https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-sorrel.pt",
}


def _resolve_weights(weights: str, cache_dir: Path | None) -> str:
    """Return a local path to the weights file, downloading if missing."""
    w = Path(weights)
    if w.exists():
        return str(w)
    if cache_dir is None:
        return weights  # defer to ultralytics (will fail if name unknown)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / w.name
    if local.exists():
        return str(local)
    url = MD_V1000_URLS.get(w.name)
    if url is None:
        return str(local)  # let ultralytics / torch error out clearly
    print(f"[MegaDetector] downloading {url} -> {local}", flush=True)
    urllib.request.urlretrieve(url, str(local))
    return str(local)


class MegaDetector:
    def __init__(
        self,
        weights: str,
        cache_dir: Path | None = None,
        device: str | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.device = device
        resolved = _resolve_weights(weights, cache_dir)
        self.model = YOLO(resolved)

    def predict(
        self,
        image_paths: Iterable[Path],
        conf: float = 0.2,
        imgsz: int = 1280,
        batch: int = 8,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> list[dict]:
        """Process in explicit chunks of ``batch`` and release tensors between
        chunks. More reliable on MPS than `stream=True`, which kept growing
        the allocator pool on Apple Silicon.

        ``progress_cb(processed, total)`` is invoked after every batch with
        the cumulative image count. One callback per batch keeps the IPC
        lightweight on large runs (3000 images → ~375 messages at batch=8).
        """
        paths = [str(p) for p in image_paths]
        total = len(paths)
        common = {
            "conf": conf,
            "imgsz": imgsz,
            "batch": batch,
            "verbose": False,
            # Class-agnostic NMS: when animal/person/vehicle boxes overlap, the
            # highest-confidence one wins across classes instead of surviving
            # alongside. MD v1000's three categories are mutually exclusive in
            # practice, so this matches the intended evaluation semantics and
            # avoids stacked mixed-label boxes on the same subject.
            "agnostic_nms": True,
        }
        if self.device:
            common["device"] = self.device

        out: list[dict] = []
        if progress_cb:
            progress_cb(0, total)
        for i in range(0, total, batch):
            chunk_paths = paths[i : i + batch]
            chunk_results = self.model.predict(chunk_paths, **common)
            # Ultralytics Results.path sometimes returns only the basename, so
            # zip against the input paths we actually passed in.
            for input_path, r in zip(chunk_paths, chunk_results):
                detections = []
                for box in r.boxes:
                    cls_id = int(box.cls)
                    detections.append(
                        {
                            "category_id": cls_id,
                            "label": LABELS.get(cls_id, str(cls_id)),
                            "conf": float(box.conf),
                            "bbox_xyxy": [float(v) for v in box.xyxy[0].tolist()],
                        }
                    )
                out.append(
                    {
                        "file": input_path,
                        "width": int(r.orig_shape[1]),
                        "height": int(r.orig_shape[0]),
                        "detections": detections,
                    }
                )
            del chunk_results
            self._release_memory()
            if progress_cb:
                progress_cb(min(i + batch, total), total)
        return out

    def _release_memory(self) -> None:
        import gc

        gc.collect()
        try:
            import torch

            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif self.device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
