# development image based on Debian image
FROM debian:bookworm-slim@sha256:903d3225acecaa272bbdd7273c6c312c2af8b73644058838d23a8c9e6e5c82cf

RUN apt-get update && apt-get install -y \
    curl \
    fuse3 \
    gcc \
    git \
    graphviz \
    less \
    libpython3-dev \
    make \
    python3 \
    python3-venv \
    rclone \
    sudo \
    vim \
    && rm -rf /var/lib/apt/lists/*

# setup user
ARG user_id=1000
RUN useradd --comment "Joe Shmoe" \
    --uid ${user_id} \
    --no-log-init \
    --create-home \
    joe \
    && echo 'joe ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER joe
ENV USER joe
ENV HOME /home/joe
ENV SHELL /bin/bash

# use home as workdir for now
WORKDIR ${HOME}

# first copy files for installing the project dependencies
# and set workdir to project/
COPY --chown=${user_id} Makefile project/
WORKDIR project/
# setup venv
RUN make .venv/bin/python3
# make commands from virtual environment available
ENV PATH="${HOME}/project/.venv/bin:${PATH}"

# install Dask
ARG DASK_VERSION="2024.9.0"
RUN pip install dask[complete]==${DASK_VERSION}

ARG JUPYTERHUB_VERSION=5.1.0
RUN pip install jupyterhub==${JUPYTERHUB_VERSION}

# install dependencies
COPY --chown=${user_id} requirements.txt ./
RUN make install

# copy project files for configuring tools
COPY --chown=${user_id} pyproject.toml ./
# copy remaining
COPY --chown=${user_id} . .

# export notebooks to have code inside docker image
RUN make export-notebooks
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="${HOME}/project/"
