#!/bin/bash
# ============================================================================
# Setup Script for NVIDIA Jetson Nano (JetPack 6.0)
# This script removes the old virtual environment and creates a fresh one
# ============================================================================

set -e  # Exit on error

echo "=========================================="
echo "Jetson Nano Environment Setup"
echo "=========================================="

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Step 1: Remove old virtual environment
echo ""
echo "[1/5] Removing old virtual environment..."
if [ -d ".venv" ]; then
    echo "  Found existing .venv directory. Removing..."
    rm -rf .venv
    echo "  ✓ Old virtual environment removed"
else
    echo "  No existing .venv found. Skipping..."
fi

# Step 2: Install system dependencies
echo ""
echo "[2/5] Installing system dependencies..."
echo "  Note: This requires sudo privileges"
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libportaudio2 \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    patchelf \
    build-essential \
    libopenmpi-dev \
    libopenmpi3

echo "  ✓ System dependencies installed"

# Step 3: Create new virtual environment
echo ""
echo "[3/5] Creating new virtual environment..."
python3 -m venv .venv
echo "  ✓ Virtual environment created"

# Step 4: Activate virtual environment
echo ""
echo "[4/5] Activating virtual environment..."
source .venv/bin/activate
echo "  ✓ Virtual environment activated"

# Step 5: Upgrade pip
echo ""
echo "[5/5] Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "  ✓ pip upgraded"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Install PyTorch manually (see requirements-jetson-nano.txt):"
echo "     wget https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl -O torch-2.4.0-cp310-cp310-linux_aarch64.whl"
echo "     pip install torch-2.4.0-cp310-cp310-linux_aarch64.whl"
echo "     pip install numpy==1.26.4  # Required by torchvision"
echo "     pip install torchvision==0.19.0 torchaudio==2.4.0 --no-deps"
echo "     pip install pillow  # Required dependency for torchvision"
echo ""
echo "  3. Fix cuDNN compatibility (REQUIRED for JetPack 6.0):"
echo "     cd .venv/lib/python3.10/site-packages/torch/lib"
echo "     ln -sf /usr/lib/aarch64-linux-gnu/libcudnn.so.9 libcudnn.so.8"
echo "     patchelf --replace-needed libcudnn.so.8 libcudnn.so.9 libtorch_python.so"
echo "     patchelf --replace-needed libcudnn.so.8 libcudnn.so.9 libtorch_cuda.so"
echo ""
echo "  4. Fix MPI compatibility (REQUIRED):"
echo "     sudo ln -sf /usr/lib/aarch64-linux-gnu/libmpi.so.40.30.2 /usr/lib/aarch64-linux-gnu/libmpi.so.20"
echo "     sudo ln -sf /usr/lib/aarch64-linux-gnu/libmpi_cxx.so.40 /usr/lib/aarch64-linux-gnu/libmpi_cxx.so.20"
echo "     sudo ldconfig"
echo ""
echo "  5. Fix CUDA library compatibility (REQUIRED):"
echo "     # Fix libcufft.so.10"
echo "     sudo ln -sf /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcufft.so.11 /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcufft.so.10"
echo "     sudo ln -sf /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcufft.so.11 /usr/lib/aarch64-linux-gnu/libcufft.so.10"
echo "     # Fix libcublas.so.10"
echo "     sudo ln -sf /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcublas.so.12 /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcublas.so.10"
echo "     sudo ln -sf /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcublas.so.12 /usr/lib/aarch64-linux-gnu/libcublas.so.10"
echo "     sudo ldconfig"
echo "     # Also add to ~/.bashrc:"
echo "     # export LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/aarch64-linux/lib:\$LD_LIBRARY_PATH"
echo ""
echo "  4. Install remaining packages:"
echo "     pip install -r requirements-jetson-nano.txt"
echo ""
