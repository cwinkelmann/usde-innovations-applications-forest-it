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

Base compose runs CPU-only. Add the `gpu` overlay on NVIDIA Linux hosts.

```bash
cd webapp

# macOS, or Linux without NVIDIA toolkit:
docker compose up --build -d

# Linux with NVIDIA Container Toolkit:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

- Webapp: http://localhost:8080
- Label Studio: http://localhost:8081 (create an account, copy the API token into the webapp's LS card, click **Save URL + token**)

## GPU support

The GPU overlay (`docker-compose.gpu.yml`) declares `deploy.resources.reservations.devices` for all NVIDIA GPUs. The app calls `torch.cuda.device_count()` at startup and spawns one worker per visible GPU, each pinned via `CUDA_VISIBLE_DEVICES`. Override worker count with `NUM_WORKERS=N` in `.env`.

- **Why it's an overlay, not the default:** before this split, Docker errored out on machines without the NVIDIA runtime (`could not select device driver "nvidia"`). The base compose is now portable; students with GPUs opt in explicitly.
- **macOS / MPS:** Docker Desktop runs containers inside a Linux VM with no Metal access — **MPS is not reachable inside the container**, regardless of compose flags. To use MPS on Mac, run the app natively (see *Dev (no Docker)* below).
- **Rootless Docker + NVIDIA:** also requires `no-cgroups = true` in `/etc/nvidia-container-runtime/config.toml`.

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

eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA4MzM3Mzk3MCwiaWF0IjoxNzc2MTczOTcwLCJqdGkiOiJiN2UwNzNiMmEzMDk0NWE3OTFjNGFmZmQzMzg4NjZiYyIsInVzZXJfaWQiOiIxIn0.fadrgVhR9AqdJf0My-l0wYIhHaqLTn8TDIBm0QYidJQ