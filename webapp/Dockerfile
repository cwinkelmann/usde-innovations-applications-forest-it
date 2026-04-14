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

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY webapp/environment-webapp.yml /app/environment-webapp.yml
COPY webapp/app /app/webapp/app

# Build the env as root. micromamba's entrypoint reads ENV_NAME to activate.
RUN micromamba create -y -n fit-webapp -f /app/environment-webapp.yml \
    && micromamba clean --all --yes

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
