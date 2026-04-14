# USDE Webapp — Design (2026-04-14)

## Goal

Dockerised NiceGUI app with three tabs — Upload, MegaDetector, MegaDetector + SpeciesNet —
running from the existing `fit-training` environment via micromamba, with a bind mount on
`/Volumes/storage/_tmp/usde`. Eventual deployment target is a multi-GPU rootless Docker host;
the app must parallelise inference across GPUs.

## Approach

- **Framework:** NiceGUI (matches HILDA conventions).
- **Environment:** single image from `environment-webapp.yml` (= `fit-training` extras + `nicegui`).
- **Base image:** `mambaorg/micromamba:1.5.8-jammy-cuda-12.1.1`.
- **Rootless Docker:** container runs as `mambauser` remapped to UID/GID 1000 (overridable).
- **Multi-GPU parallelism:** one persistent worker process per visible GPU, pinned via
  `CUDA_VISIBLE_DEVICES`. Workers spawned with `mp.get_context("spawn")` (CUDA requires spawn,
  not fork). Fallback: 1 CPU worker when no GPU is available.
- **Queue:** in-process `multiprocessing.Queue` + `Manager().dict()` status map. No Redis.
- **State:** filesystem is source of truth. `/data/uploads/<session>/` and
  `/data/outputs/<session>/<job_id>/` are scanned each time the session dropdown renders.
- **Model caching:** `MODEL_CACHE_DIR=/data/model_cache` is set as `HF_HOME`, `TORCH_HOME`,
  and `KAGGLEHUB_CACHE` so first-run downloads persist on the bind mount.

## Layout

```
webapp/
├── Dockerfile
├── docker-compose.yml
├── environment-webapp.yml
├── .dockerignore
├── .env.example
├── README.md
└── app/
    ├── main.py          # NiceGUI entry, builds 3-tab layout
    ├── config.py        # env → paths, worker count
    ├── job_manager.py   # queue + worker supervision
    ├── worker.py        # per-GPU worker loop
    ├── detectors/
    │   ├── megadetector.py   # ultralytics YOLO wrapper (P02 style)
    │   └── speciesnet.py     # SpeciesNet classifier-only wrapper (P05 style)
    └── tabs/
        ├── upload.py
        ├── megadetector.py
        └── speciesnet.py
```

## Data flow

1. **Upload tab** writes files to `/data/uploads/<session>/`. No job submitted.
2. **MD tab** and **MD+SpeciesNet tab** read the session dropdown from disk.
   Clicking Run pushes a `{id, mode, session, conf, imgsz, country?}` job onto the queue.
3. A free worker picks up the job, runs inference, writes JSON to
   `/data/outputs/<session>/<job_id>/`, and updates `status_dict[job_id]`.
4. The UI polls `jm.get_status(job_id)` at 500 ms and flips to "done" or "error".

## Error handling

- Worker exceptions → `status_dict[job_id] = {state: "error", error, traceback}`. UI surfaces the
  message. The worker continues to serve further jobs.
- No retry on failed jobs (explicit resubmit by user). Sufficient for a basic tool.
- Dead worker (process exit): out of scope for this cut; add supervisor loop in a follow-up.

## Known limits / future ideas

- No in-flight cancellation.
- No authentication.
- Session dropdown refresh is manual (button) — could auto-refresh on tab focus.
- No annotated-image rendering yet (only JSON outputs); straightforward extension.
- When the web process dies, queued jobs are lost. Move to Redis/RQ if durability matters.

## Testing

- `tests/test_detectors.py`: smoke tests that the wrapper classes import and the
  MegaDetector label map matches the ultralytics output schema. Full e2e tests require
  images + network and are deferred.
