nanugpt Docker Image

This folder contains a CUDA-enabled Docker image for running and training nanugpt on NVIDIA GPUs. The image derives from `nvcr.io/nvidia/pytorch:25.08-py3` and installs all external dependencies used by the repo, including xFormers and FlashAttention.

Contents
- Dockerfile: CUDA + PyTorch base with Python deps (einops, transformers, datasets, tiktoken, tokenizers, sentencepiece, wandb, mlflow, tqdm, matplotlib, rich, numpy, pandas, scipy) plus xFormers and FlashAttention.
- build_and_push.sh: Helper script to build and optionally push the image to a registry.

Prerequisites
- Docker 24+ and Docker Buildx
- NVIDIA Container Toolkit (for GPU support): https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
- Access to your container registry (Docker Hub or GHCR). Run `docker login` beforehand if pushing.
- Ability to pull the base image `nvcr.io/nvidia/pytorch:25.08-py3`. If your environment requires NGC auth, login with `docker login nvcr.io`.

Build
Basic local build (no push):

- `./docker/build_and_push.sh --repo <your-namespace>/nanugpt --tag <tag>`

Examples:
- `./docker/build_and_push.sh --repo username/nanugpt --tag latest`
- `./docker/build_and_push.sh --repo ghcr.io/your-org/nanugpt --tag 0.1.0`

Notes:
- The Dockerfile installs xFormers using the PyTorch CUDA wheel index. By default it uses CUDA 12.4 wheels (`cu124`). If your base image differs, override via `--cuda-index-url`:
  - CUDA 12.1: `--cuda-index-url https://download.pytorch.org/whl/cu121`
  - CUDA 11.8: `--cuda-index-url https://download.pytorch.org/whl/cu118`

Push
Build and push in one step:

- `PUSH=1 ./docker/build_and_push.sh --repo <your-namespace>/nanugpt --tag <tag>`

or with explicit flag:

- `./docker/build_and_push.sh --repo <your-namespace>/nanugpt --tag <tag> --push`

Ensure you have run `docker login` for your registry.

Use
Run with GPU and mount your working directory:

- `docker run --gpus all --ipc=host -it --rm -v "$PWD":/workspace -w /workspace <your-namespace>/nanugpt:<tag> bash`

Inside the container, Python deps are preinstalled. To train, for example:

- `python train.py --help`
- `python tokenize_dataset.py --help`

If you use Weights & Biases or MLflow, configure credentials via environment variables or mount config files/secrets as needed.

Updating
1. Edit `docker/Dockerfile` to adjust dependencies or CUDA wheel index.
2. Rebuild and push with a new tag:
   - `./docker/build_and_push.sh --repo <your-namespace>/nanugpt --tag <new-tag> --push`
3. Update your deployment manifests or scripts to use the new tag.

What’s Installed
- From `pyproject.toml`:
  - einops, tiktoken, wandb, mlflow, sentencepiece, tokenizers, transformers, datasets, tqdm, matplotlib, rich
- Additional (used by code / utilities):
  - numpy, pandas, scipy
  - xformers (for `xformers.ops.SwiGLU`)
  - flash-attn (for TinyLlama/LLaMA variants)
- Build tooling for native extensions: build-essential, ninja, cmake, python3-dev, etc.

Troubleshooting
- xFormers wheels: If install fails, your Torch/CUDA combo may not match the index. Override `--cuda-index-url` to the proper URL (e.g., cu121, cu118), or pin a compatible xFormers version.
- FlashAttention build: If a wheel isn’t available and it compiles from source, ensure adequate RAM/CPU. You may set `MAX_JOBS=<n>` during build to limit parallel compilation.
- GPU runtime: Ensure the host has recent NVIDIA drivers and that `nvidia-smi` works. Inside the container, `python -c "import torch; print(torch.cuda.is_available())"` should print `True`.

Image Naming
The default image name is `nanugpt`. Use your namespace when pushing, e.g., `username/nanugpt:latest` or `ghcr.io/your-org/nanugpt:0.1.0`.

Reproducibility Tips
- Pin tags and versions where needed in `Dockerfile`.
- Use immutable image tags in your training/serving scripts.
- Keep `docker/README.md` in sync with dependency changes in `pyproject.toml`.
