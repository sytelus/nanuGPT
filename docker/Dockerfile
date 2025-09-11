# syntax=docker/dockerfile:1.6

# Base NVIDIA PyTorch image (CUDA + cuDNN + PyTorch preinstalled)
FROM nvcr.io/nvidia/pytorch:25.08-py3

ARG DEBIAN_FRONTEND=noninteractive

# System dependencies for building native extensions and useful tooling
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
      curl \
      ca-certificates \
      pkg-config \
      python3-dev \
      ninja-build \
      cmake \
      unzip \
      jq \
      vim \
      less && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel packaging

# Important: ensure torch/compute wheels resolve correctly for CUDA in this base image.
# xFormers wheels for CUDA are typically hosted on the PyTorch index; adjust CUDA version if needed.
ARG PYTORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu124

# Core Python deps for this repo (from pyproject.toml) + extras used by the codebase
# - xformers (SwiGLU used by models)
# - flash-attn (used by TinyLlama & LLaMA variants)
# - numpy/scipy/pandas (referenced by utilities and analysis helpers)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
      einops \
      tiktoken \
      wandb \
      mlflow \
      sentencepiece \
      tokenizers \
      transformers \
      datasets \
      tqdm \
      matplotlib \
      rich \
      numpy \
      pandas \
      scipy && \
    pip install --no-cache-dir --extra-index-url ${PYTORCH_CUDA_INDEX_URL} xformers && \
    pip install --no-cache-dir flash-attn --no-build-isolation

# Optional: environment quality-of-life defaults
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create a working directory used at runtime; do not copy project by default
WORKDIR /workspace

# Print final versions for quick troubleshooting
RUN python - <<'PY'
import torch, sys
print('Python:', sys.version)
print('Torch :', torch.__version__)
print('CUDA  :', torch.version.cuda)
print('Is CUDA available:', torch.cuda.is_available())
PY

CMD ["bash"]

