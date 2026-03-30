#!/bin/bash
# Environment setup for CUDA 12.6 (required for Blackwell sm_120 / RTX PRO 6000)
set -e

pip install -U pip

# PyTorch with CUDA 12.8 — first release with sm_120 (Blackwell) support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# PyTorch Geometric (wheel URL depends on installed torch version)
TORCH=$(python -c "import torch; print(torch.__version__.split('+')[0])")
echo "Detected torch: $TORCH"
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-${TORCH}+cu126.html
pip install torch-geometric

# Everything else
pip install "pyro-ppl>=1.9.0" "numpy>=1.26.0" "pandas>=2.1.0" "scipy>=1.11.0" "tqdm>=4.66.0" lifelines scikit-learn
