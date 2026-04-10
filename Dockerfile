# fit-training Docker image
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

# Install as root so all binaries (jupyter, etc.) land in /usr/local/bin
# which is always on PATH — avoids ~/.local/bin not-found issues.
COPY pyproject.toml /build/
COPY src/ /build/src/
RUN pip install --no-cache-dir -e "/build/.[training,dev]"

ARG user_id=1000
RUN useradd --uid ${user_id} --create-home --no-log-init student \
    && echo 'student ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER student
ENV HOME=/home/student
WORKDIR /home/student/course

COPY --chown=student week1/practicals/ week1/practicals/

ENV PYTHONUNBUFFERED=1

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", \
     "--NotebookApp.token=''", "--NotebookApp.password=''"]
