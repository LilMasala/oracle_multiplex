#!/bin/bash
# Environment setup for CUDA 12.4
set -e

pip install -U pip

# PyTorch with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# PyTorch Geometric (wheel URL depends on installed torch version)
TORCH=$(python -c "import torch; print(torch.__version__.split('+')[0])")
echo "Detected torch: $TORCH"
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-${TORCH}+cu124.html
pip install torch-geometric

# Everything else
pip install "pyro-ppl>=1.9.0" "numpy>=1.26.0" "pandas>=2.1.0" "scipy>=1.11.0" "tqdm>=4.66.0"
