# fit-training Docker image
#
# Uses the official python:3.11-slim base — no conda required.
# All dependencies come from pyproject.toml extras:
#   pip install -e ".[training,dev,herdnet,labelstudio]"
#
# Build:
#   docker compose build --build-arg user_id=$(id -u)
#
# Run:
#   docker compose up

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    g++ \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

ARG user_id=1000
RUN useradd --uid ${user_id} --create-home --no-log-init student \
    && echo 'student ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER student
ENV HOME=/home/student
WORKDIR /home/student/course

# ── dependencies (cached independently of notebook changes) ─────────────────
COPY --chown=student pyproject.toml ./
COPY --chown=student src/ src/

RUN pip install --no-cache-dir -e ".[training,dev,herdnet,labelstudio]"

# ── course notebooks ─────────────────────────────────────────────────────────
COPY --chown=student week1/practicals/ week1/practicals/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/home/student/course

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", \
     "--NotebookApp.token=''", "--NotebookApp.password=''"]