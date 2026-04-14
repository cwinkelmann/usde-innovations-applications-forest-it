# USDE Wildlife Webapp

Minimal NiceGUI app with three tabs:

1. **Upload** — drop images into a named session folder under `/data/uploads/<session>/`.
2. **MegaDetector** — run MD v1000 on a session; gallery + filters + stats; Label Studio export/import with precision / recall / F1 evaluation; visual FP/FN diff; JSON / CSV / per-image CSV / per-class zip downloads.
3. **MD + SpeciesNet** — detect, then classify each animal; LS export with species pre-annotations; same downloads.

All state persists on the bind mount (`HOST_DATA_DIR` → `/data`).

## Rootless Docker

The default setup is designed for **rootless Docker**. Two things make this work:

- The container runs as **root inside**. In rootless Docker, container UID 0 maps to the host user, so files written to `/data` are owned by you on the host — no `chown` dance.
- The compose file has **no `user:` override**. Adding one (e.g. `user: "1000:1000"`) would break rootless mode because container UID 1000 maps to host UID ~100999, putting files outside your reach.

One-time host prep:

```bash
# Install rootless docker (Linux):
#   dockerd-rootless-setuptool.sh install
# Configure NVIDIA for rootless (if using GPUs):
#   sudo sed -i 's/#no-cgroups = false/no-cgroups = true/' /etc/nvidia-container-runtime/config.toml

cp webapp/.env.example webapp/.env
# Edit HOST_DATA_DIR in .env — must be a directory you own
mkdir -p "$(grep HOST_DATA_DIR webapp/.env | cut -d= -f2)"
```

## Run

```bash
cd webapp
docker compose up --build       # webapp on :8080, Label Studio on :8081
```

- Webapp: http://localhost:8080
- Label Studio: http://localhost:8081 (create an account, copy the API token into the webapp's LS card, click **Save URL + token**)

## Multi-GPU

The compose file declares `deploy.resources.reservations.devices` for all NVIDIA GPUs. The app calls `torch.cuda.device_count()` at startup and spawns one worker per visible GPU, each pinned via `CUDA_VISIBLE_DEVICES`. Override worker count with `NUM_WORKERS=N` in `.env`.

Without GPUs the reservation is ignored — the app falls back to MPS on macOS (via Docker Desktop's Metal pass-through, if available) or CPU.

## Dev (no Docker)

```bash
micromamba create -f webapp/environment-webapp.yml
micromamba activate fit-webapp
DATA_DIR=./_data python -m webapp.app.main
```

## Layout on disk

```
$HOST_DATA_DIR/
├── uploads/<session>/*.jpg              # Tab 1
├── outputs/<session>/
│   ├── md/detections.json               # MD tab — overwritten per run
│   ├── md_speciesnet/predictions.json   # SN tab — overwritten per run
│   ├── labelstudio_export.json          # MD tab LS import
│   └── labelstudio_speciesnet_export.json
├── model_cache/
│   ├── megadetector/*.pt
│   ├── kagglehub/                       # SpeciesNet weights
│   ├── huggingface/
│   └── torch/
├── labelstudio/                         # LS container's SQLite + media
└── config.json                          # LS token, export markers
```

## Why root?

Short version: rootless Docker already isolates the process from the host
root account; trying to further drop privileges *inside* the container just
breaks the bind-mount. If you're running rootful Docker in production and
want non-root, pass `--user $(id -u):$(id -g)` to compose.
