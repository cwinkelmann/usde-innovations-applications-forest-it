# fit-training Docker image
#
# Base: mambaorg/micromamba:bookworm-slim (~80 MB Debian + micromamba)
# micromamba is a drop-in for conda: same commands, faster, no base Python needed.
#
# Build:  docker compose build
# Run:    docker compose up     → JupyterLab at http://localhost:8888
# Shell:  make shell

FROM mambaorg/micromamba:bookworm-slim

ARG MAMBA_USER=mambauser

# System packages needed at build time (git for pip git-URL deps, gcc for C extensions)
# and at runtime (libgl1/libsm6/libxext6 for OpenCV).
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    g++ \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

WORKDIR /home/$MAMBA_USER/course

# Copy the environment file before creating the env
COPY --chown=$MAMBA_USER environment-demo.yml ./

# Copy source so the editable pip install inside the yml can find pyproject.toml
COPY --chown=$MAMBA_USER pyproject.toml ./
COPY --chown=$MAMBA_USER src/ src/

# Create the named conda env (name comes from environment-demo.yml: fit-demo)
RUN micromamba env create -f environment-demo.yml \
    && micromamba clean -afy

# Tell the micromamba entrypoint to activate fit-demo instead of base
ENV MAMBA_DEFAULT_ENV=fit-demo

# ── course notebooks ─────────────────────────────────────────────────────────
COPY --chown=$MAMBA_USER week1/practicals/ week1/practicals/

ENV PYTHONUNBUFFERED=1

EXPOSE 8888
CMD ["micromamba", "run", "-n", "fit-demo", \
     "jupyter", "lab", "--ip=0.0.0.0", "--no-browser", \
     "--NotebookApp.token=''", "--NotebookApp.password=''"]
