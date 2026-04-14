"""Worker process: pinned to one accelerator, loads models once, loops on the queue."""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

from .config import (
    MD_WEIGHTS,
    MODEL_CACHE_DIR,
    SPECIESNET_MODEL,
    UPLOADS_DIR,
    OUTPUTS_DIR,
    resolve_batch_size,
)


def _list_images(session_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    return sorted(p for p in session_dir.rglob("*") if p.suffix.lower() in exts)


def _set_status(status_queue, job_id: str, **fields) -> None:
    status_queue.put((job_id, fields))


def worker_main(gpu_idx: int, device: str, job_queue, status_queue) -> None:
    """
    gpu_idx: CUDA device index (-1 means no CUDA pinning). Unused for MPS/CPU.
    device:  "cuda" | "mps" | "cpu" — passed to ultralytics/torch.
    """
    # CUDA pinning must happen before torch/ultralytics import.
    if device == "cuda" and gpu_idx >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    # Point HF/Torch caches at the persistent mount so first-run downloads stick.
    os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(MODEL_CACHE_DIR / "torch"))
    os.environ.setdefault("KAGGLEHUB_CACHE", str(MODEL_CACHE_DIR / "kagglehub"))
    # On MPS, PyTorch pre-allocates a big slab by default. Letting it grow from
    # zero dramatically reduces peak resident memory on laptops.
    if device == "mps":
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    from .detectors.megadetector import MegaDetector

    md = MegaDetector(
        MD_WEIGHTS, cache_dir=MODEL_CACHE_DIR / "megadetector", device=device
    )
    default_batch = resolve_batch_size(device)
    snet = None  # lazy-loaded on first md+speciesnet job

    while True:
        job = job_queue.get()
        if job is None:
            return
        job_id = job["id"]
        try:
            _set_status(
                status_queue,
                job_id,
                state="running",
                progress=0.0,
                worker=f"{device}:{gpu_idx}" if device == "cuda" else device,
            )
            session_dir = UPLOADS_DIR / job["session"]
            images = _list_images(session_dir)
            if not images:
                raise RuntimeError(f"No images found in {session_dir}")

            # Results are keyed by MODE, not job id: re-running overwrites the
            # previous run's JSON for that session+mode. Older job_id folders
            # are no longer created.
            mode_key = "md" if job["mode"] == "md" else "md_speciesnet"
            out_dir = OUTPUTS_DIR / job["session"] / mode_key
            out_dir.mkdir(parents=True, exist_ok=True)

            # Progress cap for MD: 1.0 for MD-only runs, 0.5 for combined
            # runs (SpeciesNet claims the second half of the bar).
            md_cap = 0.5 if job["mode"] == "md+speciesnet" else 1.0

            def md_progress(processed: int, total: int) -> None:
                frac = processed / max(1, total)
                _set_status(
                    status_queue,
                    job_id,
                    stage="MegaDetector",
                    processed=processed,
                    total=total,
                    progress=frac * md_cap,
                )

            md_results = md.predict(
                images,
                conf=job.get("conf", 0.2),
                imgsz=job.get("imgsz", 1280),
                batch=job.get("batch", default_batch),
                progress_cb=md_progress,
            )
            (out_dir / "detections.json").write_text(json.dumps(md_results, indent=2))
            _set_status(status_queue, job_id, progress=md_cap)

            result_payload = {
                "mode": job["mode"],
                "num_images": len(images),
                "out_dir": str(out_dir),
            }

            if job["mode"] == "md+speciesnet":
                if snet is None:
                    from .detectors.speciesnet import SpeciesNetClassifier

                    snet = SpeciesNetClassifier(SPECIESNET_MODEL)

                def snet_progress(processed: int, total: int) -> None:
                    frac = processed / max(1, total)
                    _set_status(
                        status_queue,
                        job_id,
                        stage="SpeciesNet",
                        processed=processed,
                        total=total,
                        progress=0.5 + frac * 0.5,
                    )

                merged = snet.predict(
                    md_results,
                    country=job.get("country"),
                    batch_size=job.get("batch", default_batch),
                    progress_cb=snet_progress,
                )
                (out_dir / "predictions.json").write_text(json.dumps(merged, indent=2))

            _set_status(
                status_queue, job_id, state="done", progress=1.0, result=result_payload
            )
        except Exception as e:
            _set_status(
                status_queue,
                job_id,
                state="error",
                error=str(e),
                traceback=traceback.format_exc(),
            )
