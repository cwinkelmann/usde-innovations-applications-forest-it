# JupyterHub Setup — Forest IT Module

## Overview

This sets up a multi-user JupyterHub for the Forest IT course on an Ubuntu server with 8 GPUs and rootless Docker. The architecture:

```
Student's browser
  → JupyterHub (Python process in your conda env on the host, port 8000)
      → Rootless Docker daemon (running as your user)
          → Container per student (isolated JupyterLab + 1 GPU each)
```

JupyterHub is **not** containerized — it runs directly in a conda environment under your user account. When a student logs in, JupyterHub tells Docker to spin up an isolated container for that student. Each container:

- Runs JupyterLab with a full ML stack (PyTorch, ultralytics, etc.)
- Sees exactly one GPU via `NVIDIA_VISIBLE_DEVICES`
- Has a persistent workspace bind-mounted from the host
- Has read-only access to a shared data directory
- Has read-write access to a shared results directory
- Is destroyed on logout (but the student's files persist on the host)

## Prerequisites

Before starting, confirm the following on the server:

```bash
# Rootless Docker is running
docker info 2>/dev/null | grep -i rootless

# NVIDIA drivers are installed
nvidia-smi

# NVIDIA Container Toolkit is installed and configured for rootless
# This line must exist in /etc/nvidia-container-runtime/config.toml:
#   [nvidia-container-cli]
#   no-cgroups = true
# If it doesn't, ask a sysadmin to add it (one-time, single-line change).

# Conda is available
conda --version
```

## Directory Structure

All paths are relative to your home directory (`~`). Create them before proceeding.

```
~/jupyterhub/
├── jupyterhub_config.py      # Hub configuration
├── Dockerfile.student         # Student container image definition
├── student-requirements.txt   # Python packages for student image
├── docker-compose.yml         # (optional) for running hub + proxy
└── README.md                  # Quick-start reference

~/hub-users/                   # Per-student persistent workspaces
│   ├── alice/                 # Created automatically on first login
│   ├── bob/
│   └── .../

~/hub-shared-data/             # Read-only shared data pool
│   ├── drone-imagery/
│   ├── model-weights/
│   └── sample-datasets/

~/hub-shared-results/          # Writable shared output folder
```

```bash
mkdir -p ~/jupyterhub ~/hub-users ~/hub-shared-data ~/hub-shared-results
```

## Step 1: Create the Conda Environment

```bash
conda create -n jupyterhub python=3.11 -y
conda activate jupyterhub
pip install jupyterhub jupyterlab dockerspawner docker
```

## Step 2: Build the Student Docker Image

### student-requirements.txt

```txt
jupyterlab
jupyterhub
torch
torchvision
torchaudio
ultralytics
marimo
rasterio
geopandas
matplotlib
seaborn
scikit-learn
scipy
pandas
numpy
opencv-python-headless
Pillow
tqdm
ipywidgets
```

### Dockerfile.student

```dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    libgdal-dev \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

COPY student-requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Create non-root user inside the container
RUN useradd -m -s /bin/bash student
USER student
WORKDIR /home/student

# JupyterHub single-user entry point
CMD ["jupyterhub-singleuser", "--ip=0.0.0.0"]
```

### Build

```bash
cd ~/jupyterhub
docker build -t forestit-student -f Dockerfile.student .
```

Verify GPU access works inside the image:

```bash
docker run --rm --gpus '"device=0"' forestit-student python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Step 3: Configure JupyterHub

### jupyterhub_config.py

```python
import os
import hashlib
import docker

# ──────────────────────────────────────────────
# Network
# ──────────────────────────────────────────────
c.JupyterHub.bind_url = 'http://0.0.0.0:8000'
c.JupyterHub.hub_ip = '0.0.0.0'

# The hub must be reachable from inside containers.
# For rootless Docker, find the bridge IP automatically.
c.JupyterHub.hub_connect_ip = os.popen(
    "ip addr show docker0 2>/dev/null | grep 'inet ' | awk '{print $2}' | cut -d/ -f1"
).read().strip() or '172.17.0.1'

# ──────────────────────────────────────────────
# Spawner: DockerSpawner
# ──────────────────────────────────────────────
c.JupyterHub.spawner_class = 'dockerspawner.DockerSpawner'
c.DockerSpawner.image = 'forestit-student'
c.DockerSpawner.remove = True          # destroy container on logout
c.DockerSpawner.name_template = 'jupyter-{username}'

# Rootless Docker socket
uid = os.getuid()
rootless_socket = f'/run/user/{uid}/docker.sock'
if os.path.exists(rootless_socket):
    c.DockerSpawner.docker_socket = f'unix://{rootless_socket}'

# Use the internal Docker network so containers can reach the hub
c.DockerSpawner.network_name = 'jupyterhub-network'
c.DockerSpawner.use_internal_ip = True

# ──────────────────────────────────────────────
# Volumes
# ──────────────────────────────────────────────
HOME = os.path.expanduser('~')

c.DockerSpawner.volumes = {
    # Per-student persistent workspace
    os.path.join(HOME, 'hub-users', '{username}'): {
        'bind': '/home/student/work',
        'mode': 'rw',
    },
    # Shared data (read-only)
    os.path.join(HOME, 'hub-shared-data'): {
        'bind': '/home/student/shared-data',
        'mode': 'ro',
    },
    # Shared results (writable)
    os.path.join(HOME, 'hub-shared-results'): {
        'bind': '/home/student/shared-results',
        'mode': 'rw',
    },
}

c.DockerSpawner.notebook_dir = '/home/student/work'

# ──────────────────────────────────────────────
# GPU Assignment
# ──────────────────────────────────────────────
GPU_COUNT = 8

def pre_spawn_hook(spawner):
    """Assign one GPU per student (deterministic by username)."""
    username = spawner.user.name

    # Create host-side workspace if it doesn't exist
    user_dir = os.path.join(HOME, 'hub-users', username)
    os.makedirs(user_dir, exist_ok=True)

    # Deterministic GPU assignment
    gpu_id = int(hashlib.md5(username.encode()).hexdigest(), 16) % GPU_COUNT

    spawner.extra_host_config = {
        'device_requests': [
            docker.types.DeviceRequest(
                device_ids=[str(gpu_id)],
                capabilities=[['gpu']],
            )
        ],
    }
    spawner.environment['NVIDIA_VISIBLE_DEVICES'] = str(gpu_id)

c.DockerSpawner.pre_spawn_hook = pre_spawn_hook

# ──────────────────────────────────────────────
# Resource Limits
# ──────────────────────────────────────────────
c.DockerSpawner.mem_limit = '32G'
c.DockerSpawner.cpu_limit = 8

# Auto-cull idle containers after 1 hour
c.JupyterHub.services = [
    {
        'name': 'cull-idle',
        'admin': True,
        'command': [
            'python3', '-m', 'jupyterhub_idle_culler',
            '--timeout=3600',
            '--cull-every=300',
        ],
    }
]

# ──────────────────────────────────────────────
# Authentication
# ──────────────────────────────────────────────
from jupyterhub.auth import DummyAuthenticator

c.JupyterHub.authenticator_class = DummyAuthenticator
c.DummyAuthenticator.password = 'forestIT2026'

# Optional: restrict to known usernames
# c.Authenticator.allowed_users = {
#     'student01', 'student02', 'student03',
#     # ... one per student
# }

# ──────────────────────────────────────────────
# Idle culler dependency
# ──────────────────────────────────────────────
# Install with: pip install jupyterhub-idle-culler
```

## Step 4: Create the Docker Network

DockerSpawner needs a dedicated Docker network so that containers can reach the hub process on the host.

```bash
docker network create jupyterhub-network
```

## Step 5: Install the Idle Culler (optional but recommended)

This automatically shuts down containers that have been idle for more than 1 hour, freeing GPU memory.

```bash
conda activate jupyterhub
pip install jupyterhub-idle-culler
```

## Step 6: Populate Shared Data

Place your course materials into the shared data directory:

```bash
# Example structure
~/hub-shared-data/
├── drone-imagery/
│   ├── sample-tiles/          # Small subset for practicals
│   └── README.md              # Description of data
├── model-weights/
│   ├── megadetector_v5.pt
│   └── yolov8n.pt
├── sample-datasets/
│   └── galapagos-iguanas-demo/
└── notebooks/
    ├── practical-01-intro.ipynb
    ├── practical-02-camera-traps.ipynb
    └── practical-03-megadetector.ipynb
```

Students see this at `/home/student/shared-data/` inside their container (read-only). They can copy notebooks to their own workspace to edit them.

## Step 7: Launch

```bash
# Make sure rootless Docker is running
systemctl --user start docker

# Activate the environment
conda activate jupyterhub

# Start JupyterHub
cd ~/jupyterhub
jupyterhub -f jupyterhub_config.py
```

Students navigate to `http://<server-ip>:8000` in their browser.

### Running in the Background

For persistent operation (survives SSH disconnects):

```bash
# Option A: tmux/screen
tmux new -s jupyterhub
conda activate jupyterhub
cd ~/jupyterhub
jupyterhub -f jupyterhub_config.py

# Option B: nohup
nohup jupyterhub -f jupyterhub_config.py > ~/jupyterhub/hub.log 2>&1 &
```

## Step 8: TLS / HTTPS (if needed)

If students access the server over the open internet (not behind campus VPN), add TLS. The simplest approach is a reverse proxy with Let's Encrypt:

```bash
# Install caddy (automatic HTTPS)
# In Caddyfile:
# yourdomain.example.com {
#     reverse_proxy localhost:8000
# }
```

Alternatively, JupyterHub supports TLS natively:

```python
c.JupyterHub.ssl_key = '/path/to/privkey.pem'
c.JupyterHub.ssl_cert = '/path/to/fullchain.pem'
c.JupyterHub.bind_url = 'https://0.0.0.0:443'
```

## Troubleshooting

### Container can't reach JupyterHub

Check that `hub_connect_ip` resolves to an IP reachable from inside Docker containers:

```bash
# From host, check the bridge IP
ip addr show docker0

# Test from inside a container
docker run --rm --network jupyterhub-network forestit-student \
    python3 -c "import urllib.request; urllib.request.urlopen('http://<bridge-ip>:8081/hub/api')"
```

### GPU not visible inside container

```bash
# Test GPU passthrough directly
docker run --rm --gpus '"device=0"' forestit-student nvidia-smi
```

If this fails, check that `/etc/nvidia-container-runtime/config.toml` has `no-cgroups = true` (requires sysadmin for rootless Docker).

### Student's files disappear

Files are only persisted if they're saved inside `/home/student/work/` (which maps to `~/hub-users/<username>/` on the host). Files saved elsewhere in the container are lost on logout because `remove = True`.

### Hash collisions in GPU assignment

With 8 GPUs and ~20 students, some students will share a GPU. This is fine for inference and light training. If you need guaranteed 1:1 assignment during heavy practicals, replace the hash function with an explicit mapping:

```python
GPU_MAP = {
    'student01': '0', 'student02': '1', 'student03': '2',
    # ...
}
# In pre_spawn_hook:
gpu_id = GPU_MAP.get(username, str(hash(username) % GPU_COUNT))
```

## Quick Reference

| Action | Command |
|--------|---------|
| Start hub | `conda activate jupyterhub && cd ~/jupyterhub && jupyterhub -f jupyterhub_config.py` |
| Rebuild student image | `cd ~/jupyterhub && docker build -t forestit-student -f Dockerfile.student .` |
| List running containers | `docker ps --filter name=jupyter-` |
| Stop all student containers | `docker stop $(docker ps -q --filter name=jupyter-)` |
| Check GPU usage | `nvidia-smi` |
| View hub logs | Check terminal or `~/jupyterhub/hub.log` |
| Add a package to student env | Edit `student-requirements.txt`, rebuild image, students re-login |