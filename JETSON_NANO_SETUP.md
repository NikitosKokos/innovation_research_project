# Jetson Nano Setup Guide (JetPack 6.0)

## ⚠️ Known Compatibility Issue

**PyTorch 2.4.0 for JetPack 6.0 has a cuDNN version mismatch:**
- PyTorch wheel expects: `libcudnn.so.8` (with version symbols)
- JetPack 6.0 provides: `libcudnn.so.9` (different version symbols)
- This causes: `ImportError: version 'libcudnn.so.8' not found`

## ✅ Recommended Solutions

### Solution 1: Use NVIDIA Docker Container (Easiest)

```bash
# Pull pre-configured PyTorch container
docker pull dustynv/l4t-pytorch:r36.2.0

# Run container with GPU access
docker run --runtime=nvidia -it --rm \
    --network host \
    -v /home/ailab/Desktop/innovation_research_project:/workspace \
    dustynv/l4t-pytorch:r36.2.0
```

### Solution 2: Try PyTorch 2.2.0 or 2.3.0

Check NVIDIA forums for wheels that may have better cuDNN 9 compatibility:
- https://forums.developer.nvidia.com/t/pytorch-for-jetpack-6-0/275200

### Solution 3: Build PyTorch from Source

This is time-consuming (several hours) but guarantees compatibility:

```bash
# Install build dependencies
sudo apt-get install -y build-essential cmake ninja-build

# Clone and build PyTorch
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout v2.4.0
python setup.py install
```

## 📋 Manual Setup Steps (If Not Using Docker)

### 1. Remove Old Virtual Environment

```bash
cd /home/ailab/Desktop/innovation_research_project
rm -rf .venv
```

### 2. Install System Dependencies

```bash
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
    build-essential
```

### 3. Create New Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### 4. Install PyTorch (Manual)

```bash
# Download NVIDIA PyTorch wheel
wget https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl -O torch-2.4.0-cp310-cp310-linux_aarch64.whl

# Install PyTorch
pip install torch-2.4.0-cp310-cp310-linux_aarch64.whl

# Install numpy first (required by torchvision)
pip install numpy==1.26.4

# Install torchvision and torchaudio
pip install torchvision==0.19.0 torchaudio==2.4.0 --no-deps
pip install pillow
```

### 5. Apply cuDNN Compatibility Fixes (Partial Workaround)

```bash
cd .venv/lib/python3.10/site-packages/torch/lib

# Create symlink
ln -sf /usr/lib/aarch64-linux-gnu/libcudnn.so.9 libcudnn.so.8

# Patch binary dependencies
patchelf --replace-needed libcudnn.so.8 libcudnn.so.9 libtorch_python.so
patchelf --replace-needed libcudnn.so.8 libcudnn.so.9 libtorch_cuda.so
```

**Note:** These fixes may not fully resolve the version symbol requirement. If you still get errors, use Solution 1 (Docker) or Solution 2 (different PyTorch version).

### 6. Install Remaining Packages

```bash
pip install -r requirements-jetson-nano.txt
```

## 🔍 Verification

```bash
source .venv/bin/activate
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## 📚 Additional Resources

- [NVIDIA Jetson PyTorch Installation Guide](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
- [Jetson Containers (dusty-nv)](https://github.com/dusty-nv/jetson-containers)
- [NVIDIA Developer Forums - PyTorch for JetPack 6.0](https://forums.developer.nvidia.com/t/pytorch-for-jetpack-6-0/275200)
