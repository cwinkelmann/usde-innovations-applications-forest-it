"""Runtime configuration read from env vars.

All paths inside the container are under /data (bind-mounted from
/Volumes/storage/_tmp/usde on the host). The host path is not referenced
here — only the container-side layout matters for the running process.
"""
from __future__ import annotations

import os
from pathlib import Path


DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUTS_DIR = DATA_DIR / "outputs"
MODEL_CACHE_DIR = DATA_DIR / "model_cache"

MD_WEIGHTS = os.environ.get("MD_WEIGHTS", "md_v1000.0.0-larch.pt")
SPECIESNET_MODEL = os.environ.get(
    "SPECIESNET_MODEL", "kaggle:google/speciesnet/pyTorch/v4.0.2a/1"
)

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8080"))

# DEVICE=auto → cuda if available, else mps (Apple Silicon), else cpu.
_device_raw = os.environ.get("DEVICE", "auto")

# NUM_WORKERS=auto → one worker per visible CUDA GPU. MPS / CPU get a single
# worker because those back-ends have one shared device.
_num_workers_raw = os.environ.get("NUM_WORKERS", "auto")


def resolve_device() -> str:
    if _device_raw != "auto":
        return _device_raw
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def resolve_num_workers() -> int:
    if _num_workers_raw != "auto":
        return max(1, int(_num_workers_raw))
    try:
        import torch

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            return n if n > 0 else 1
    except Exception:
        pass
    return 1  # MPS or CPU: single worker shares the one device


def resolve_batch_size(device: str | None = None) -> int:
    """Sensible per-device default.

    - cuda: 16 (plenty of VRAM on modern GPUs, makes throughput usable)
    - mps:  2  (Metal runs OOM / stalls at larger batches on most Macs)
    - cpu:  8  (moderate; big batches don't help on CPU)
    """
    override = os.environ.get("BATCH_SIZE")
    if override:
        return max(1, int(override))
    if device is None:
        device = resolve_device()
    if device == "cuda":
        return 16
    if device == "mps":
        return 2
    return 8


def ensure_dirs() -> None:
    for d in (UPLOADS_DIR, OUTPUTS_DIR, MODEL_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
