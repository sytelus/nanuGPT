#! /bin/bash

set -eu -o pipefail # fail if any command failes, log all commands, -o xtrace

conda create -y -n zentorch python=3.11.9 anaconda
conda activate zentorch

# 1) Install a matching PyTorch (CPU build)
# If you already have CUDA on this machine that’s fine—zentorch is a CPU backend.
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# 2) Install the PyTorch ZenDNN plugin
# (wheel published on PyPI; for pinned combos see the GitHub releases page)
pip install zentorch

# 3) Quick sanity check
python - <<'PY'
import torch, zentorch
print("torch:", torch.__version__)
print("zentorch:", zentorch.__version__)
print("BF16 available on CPU?", torch.tensor([1.], dtype=torch.bfloat16).device == torch.device("cpu"))
PY
