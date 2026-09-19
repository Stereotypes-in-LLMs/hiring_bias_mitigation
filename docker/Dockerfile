# Default base image: NVIDIA NGC PyTorch container, which ships CUDA/cuDNN and a matching
# torch build for both x86_64 and aarch64 (Grace Blackwell / DGX Spark, SBSA). Override with
# --build-arg BASE_IMAGE=... for a different CUDA/driver combo; tags are at
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE}

# Do NOT `pip install torch` on top of this image. The NGC base ships a torch build tuned for
# this exact CUDA and architecture; a generic PyPI wheel silently breaks GPU support, and on
# aarch64 there often is no working wheel at all.

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e . --no-deps

COPY configs/ configs/
COPY scripts/ scripts/
COPY data/ data/
COPY tests/ tests/

# Defaults for a bare `docker run`; docker-compose.yml overrides these to point at the
# bind-mounted /models drive.
ENV PYTHONUNBUFFERED=1 \
    HBM_OUTPUT_ROOT=/models \
    HF_HOME=/models/hf-cache \
    WANDB_DIR=/models/wandb \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

CMD ["bash"]
