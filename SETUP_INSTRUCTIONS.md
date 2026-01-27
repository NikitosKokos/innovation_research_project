# Complete Setup Instructions for Jetson Nano (JetPack 6.0)

## Overview

This guide provides a complete, tested setup procedure for installing PyTorch and all dependencies on NVIDIA Jetson Nano running JetPack 6.0 with cuDNN 9.

## Quick Start

Run the automated setup script:

```bash
cd /home/ailab/Desktop/innovation_research_project
./COMPLETE_SETUP_JETSON_NANO.sh
```

This script will:
1. Clean up any existing broken installations
2. Install all system dependencies
3. Create a fresh virtual environment
4. Install PyTorch 2.2.0 (best cuDNN 9 compatibility)
5. Apply all library compatibility fixes
6. Install remaining Python packages
7. Verify the installation

## Manual Installation (If Script Fails)

### Step 1: Install System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    libopenblas-dev \
    libopenblas-base \
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
    libopenmpi-dev \
    libomp-dev \
    wget \
    curl
```

### Step 2: Create Virtual Environment

```bash
cd /home/ailab/Desktop/innovation_research_project
rm -rf .venv  # Remove old venv if exists
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### Step 3: Install NumPy

```bash
pip install --no-cache-dir "numpy<2.0"
```

### Step 4: Install PyTorch 2.2.0 (Recommended for cuDNN 9)

```bash
# Download PyTorch 2.2.0 wheel
wget https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.2.0a0+6a974be.nv23.11-cp310-cp310-linux_aarch64.whl -O torch-2.2.0-cp310-cp310-linux_aarch64.whl

# Install PyTorch
pip install --no-cache-dir torch-2.2.0-cp310-cp310-linux_aarch64.whl

# Install TorchVision and TorchAudio
pip install --no-cache-dir torchvision==0.17.0 torchaudio==2.2.0 --no-deps
pip install --no-cache-dir pillow
```

### Step 5: Apply Library Compatibility Fixes

```bash
# Run the library fix script
./QUICK_FIX_ALL_LIBRARIES.sh

# Or manually:
sudo ln -sf /usr/lib/aarch64-linux-gnu/libcudnn.so.9 /usr/lib/aarch64-linux-gnu/libcudnn.so.8
sudo ln -sf /usr/lib/aarch64-linux-gnu/libmpi.so.40.30.2 /usr/lib/aarch64-linux-gnu/libmpi.so.20
sudo ln -sf /usr/lib/aarch64-linux-gnu/libmpi_cxx.so.40 /usr/lib/aarch64-linux-gnu/libmpi_cxx.so.20
sudo ln -sf /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcufft.so.11 /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcufft.so.10
sudo ln -sf /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcublas.so.12 /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcublas.so.10
sudo ln -sf /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudart.so.12 /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudart.so.10
sudo ln -sf /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudart.so.12 /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudart.so.10.2
sudo ldconfig

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/aarch64-linux/lib:$LD_LIBRARY_PATH
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/aarch64-linux/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Step 6: Fix cuDNN in PyTorch Installation

```bash
source .venv/bin/activate
VENV_TORCH_LIB="$PWD/.venv/lib/python3.10/site-packages/torch/lib"
SYSTEM_CUDNN="/usr/lib/aarch64-linux-gnu/libcudnn.so.9"

# Create symlink
ln -sf "$SYSTEM_CUDNN" "$VENV_TORCH_LIB/libcudnn.so.8"

# Patch binaries
for lib in "$VENV_TORCH_LIB"/*.so; do
    if [ -f "$lib" ] && readelf -d "$lib" 2>/dev/null | grep -q "libcudnn.so.8"; then
        patchelf --replace-needed libcudnn.so.8 libcudnn.so.9 "$lib" 2>/dev/null || true
    fi
done
```

### Step 7: Install Remaining Packages

```bash
source .venv/bin/activate
pip install --no-cache-dir -r requirements-jetson-nano.txt
```

### Step 8: Verify Installation

```bash
source .venv/bin/activate
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"
```

## Troubleshooting

### PyTorch Import Errors

If you get `AttributeError: module 'torch' has no attribute '__version__'`:
- PyTorch installation is corrupted
- Run the complete setup script to reinstall everything

### cuDNN Version Errors

If you get `libcudnn.so.8: version 'libcudnn.so.8' not found`:
- This is a version symbol mismatch
- PyTorch 2.2.0 has better cuDNN 9 compatibility than 2.4.0
- Ensure all library fixes are applied (Step 5)

### CUDA Not Available

If `torch.cuda.is_available()` returns `False`:
- Check that library fixes are applied
- Verify `LD_LIBRARY_PATH` is set correctly
- Run `sudo ldconfig` to update library cache

## Alternative: Docker Container

If installation continues to fail, use NVIDIA's pre-configured Docker container:

```bash
docker pull dustynv/l4t-pytorch:r36.2.0
```

This container has PyTorch pre-configured for JetPack 6.0.

## References

- [NVIDIA PyTorch for Jetson Platform](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
- [NVIDIA Developer Forums - PyTorch for JetPack 6.0](https://forums.developer.nvidia.com/t/pytorch-for-jetpack-6-0/275200)
- [NVIDIA cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/)
