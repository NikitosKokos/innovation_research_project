# Option 2: PyTorch 2.2.0 or 2.3.0 for JetPack 6.0 (cuDNN 9 Compatible)

This guide provides step-by-step instructions to install PyTorch 2.2.0 or 2.3.0 on JetPack 6.0, which may have better cuDNN 9 compatibility than PyTorch 2.4.0.

## Prerequisites

- JetPack 6.0 (L4T R36.x) installed
- Python 3.10
- Virtual environment activated (`.venv`)

## Step-by-Step Installation

### Step 1: Remove Current PyTorch Installation

```bash
cd /home/ailab/Desktop/innovation_research_project
source .venv/bin/activate

# Uninstall current PyTorch
pip uninstall -y torch torchvision torchaudio
```

### Step 2: Choose Your PyTorch Version

**Option A: PyTorch 2.3.0 (Recommended - Verified Available)**
```bash
# Download PyTorch 2.3.0 wheel from NVIDIA Box
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-2.3.0-cp310-cp310-linux_aarch64.whl

# Install PyTorch 2.3.0
pip install torch-2.3.0-cp310-cp310-linux_aarch64.whl
```

**Option B: Check Available PyTorch Versions**
```bash
# List available wheels in NVIDIA repository
curl -s https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/ | grep -o 'torch-[0-9].*\.whl' | sort -V

# Then download the specific version you want
# Example for a different 2.4.0 build:
# wget https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+3bcc3cddb5.nv24.07.16234504-cp310-cp310-linux_aarch64.whl
```

**Option B: PyTorch 2.3.0 (Alternative)**
```bash
# Download PyTorch 2.3.0 wheel
wget https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0-cp310-cp310-linux_aarch64.whl -O torch-2.3.0-cp310-cp310-linux_aarch64.whl

# Install PyTorch 2.3.0
pip install torch-2.3.0-cp310-cp310-linux_aarch64.whl
```

### Step 3: Install NumPy (Required by torchvision)

```bash
pip install numpy==1.26.4
```

### Step 4: Install TorchVision and TorchAudio

**Important:** Standard PyPI torchvision may not be compatible. You have two options:

**Option A: Install from PyPI (May work, but may downgrade PyTorch)**
```bash
# For PyTorch 2.2.0
pip install torchvision==0.17.0 torchaudio==2.2.0 --no-deps

# For PyTorch 2.3.0
pip install torchvision==0.18.0 torchaudio==2.3.0 --no-deps

# Install required dependencies
pip install pillow
```

**Option B: Extract from NVIDIA Docker Container (More Reliable)**
```bash
# Pull the container
docker pull dustynv/l4t-pytorch:r36.2.0

# Extract torchvision and torchaudio wheels from container
docker run --rm dustynv/l4t-pytorch:r36.2.0 find /opt -name "torchvision*.whl" -o -name "torchaudio*.whl" > /tmp/wheels.txt

# Copy wheels out (adjust paths as needed)
# Then install: pip install /path/to/extracted/torchvision.whl
```

### Step 5: Apply cuDNN Compatibility Fixes

Even with PyTorch 2.2.0/2.3.0, you may still need these fixes:

```bash
cd .venv/lib/python3.10/site-packages/torch/lib

# Create symlink
ln -sf /usr/lib/aarch64-linux-gnu/libcudnn.so.9 libcudnn.so.8

# Patch binary dependencies
patchelf --replace-needed libcudnn.so.8 libcudnn.so.9 libtorch_python.so
patchelf --replace-needed libcudnn.so.8 libcudnn.so.9 libtorch_cuda.so
```

### Step 6: Verify Installation

```bash
source .venv/bin/activate
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

### Step 7: Test with Your Application

```bash
python project/real_time_voice_conversion/app.py
```

## Troubleshooting

### If you still get cuDNN version errors:

1. **Check which PyTorch version you installed:**
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

2. **Verify cuDNN fix was applied:**
   ```bash
   readelf -d .venv/lib/python3.10/site-packages/torch/lib/libtorch_python.so | grep NEEDED | grep cudnn
   ```
   Should show: `libcudnn.so.9`

3. **Try the other PyTorch version** (2.2.0 if you tried 2.3.0, or vice versa)

### If torchvision installation fails:

- Use `--no-deps` flag and install dependencies manually
- Consider building torchvision from source (time-consuming)
- Extract wheels from NVIDIA Docker container

## Expected Results

- ✅ PyTorch imports without errors
- ✅ `torch.cuda.is_available()` returns `True`
- ✅ Your Seed-VC model loads successfully
- ✅ No cuDNN version errors

## Next Steps

Once PyTorch is working:
1. Install remaining packages: `pip install -r requirements-jetson-nano.txt`
2. Test your real-time voice conversion app
3. If issues persist, document the specific error and consider Option 1 (Docker)
