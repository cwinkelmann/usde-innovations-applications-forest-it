# syntax=docker/dockerfile:1.6
#
# USDE Wildlife Webapp — rootless-Docker friendly.
#
# Rationale for running as root inside the container:
#   In rootless Docker, the container's UID 0 maps to the host user that
#   started the daemon (your own user). Any file the container writes to a
#   bind-mounted volume is therefore owned by that host user, with no UID
#   juggling required. Running as a non-root container UID (e.g. 1000) in
#   rootless mode instead maps to host UID ~100999 (inside the user's
#   subuid range), and the host user cannot read those files without
#   chown'ing through `newuidmap`.
#
#   For rootful Docker deployments that genuinely need a non-root user,
#   pass `--user $(id -u):$(id -g)` / compose's `user:` key — running as
#   root here doesn't force anything.

ARG MAMBA_IMAGE=mambaorg/micromamba:1.5.8-jammy-cuda-12.1.1
FROM ${MAMBA_IMAGE}

USER root
RUN mkdir -p /app /data

# Runtime libs for OpenCV (pulled in by ultralytics → cv2):
#   libgl1          → libGL.so.1
#   libglib2.0-0    → libgobject-2.0.so.0
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Layer cache strategy:
# 1. COPY only metadata + a stub src/ (just enough for `pip install -e .`
#    in the env file to resolve the package layout).
# 2. micromamba create — heavy, ~5 min, cached on every subsequent build
#    where pyproject.toml + environment-webapp.yml are unchanged.
# 3. COPY the real src + webapp/app on top — these are the layers that
#    invalidate on each code edit, but they're milliseconds.
# The stub is overwritten by the real src in step 3; the editable install's
# .pth still points at /app/src/wildlife_detection so imports just work.
COPY pyproject.toml /app/pyproject.toml
COPY webapp/environment-webapp.yml /app/environment-webapp.yml

# Empty package stub: setuptools.find_packages(where=["src"]) walks this
# directory at install time. An empty __init__.py is enough for the
# editable install to register the package.
RUN mkdir -p /app/src/wildlife_detection \
 && touch /app/src/wildlife_detection/__init__.py

# Build the env as root. micromamba's entrypoint reads ENV_NAME to activate.
RUN micromamba create -y -n fit-webapp -f /app/environment-webapp.yml \
    && micromamba clean --all --yes

# Real source — overlays the stub. No reinstall needed; the editable
# .pth from the previous step already points here.
COPY src /app/src
COPY webapp/app /app/webapp/app


# Ensure we stay as root at runtime (mambaorg/micromamba defaults to mambauser).
USER root

ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV ENV_NAME=fit-webapp
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/data
ENV PORT=8080
ENV HOST=0.0.0.0

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["python", "-m", "webapp.app.main"]
