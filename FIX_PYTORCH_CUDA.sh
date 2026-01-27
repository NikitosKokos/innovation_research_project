#!/bin/bash
# ============================================================================
# Fix PyTorch Installation - Replace CPU-only with CUDA-enabled NVIDIA Wheel
# ============================================================================

set -e

cd /home/ailab/Desktop/innovation_research_project
source .venv/bin/activate

echo "Checking current PyTorch installation..."

# Check if PyTorch has CUDA support
CUDA_COMPILED=$(python -c "import torch; import torch._C; print(hasattr(torch._C, '_cuda'))" 2>/dev/null || echo "False")

if [ "$CUDA_COMPILED" = "True" ]; then
    echo "✓ PyTorch already has CUDA support"
    python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
    exit 0
fi

echo "⚠ PyTorch is CPU-only. Reinstalling NVIDIA CUDA-enabled wheel..."

# Uninstall CPU-only PyTorch
pip uninstall -y torch torchvision torchaudio

# Download NVIDIA PyTorch 2.4.0 wheel if not present
if [ ! -f "torch-2.4.0-cp310-cp310-linux_aarch64.whl" ]; then
    echo "Downloading PyTorch 2.4.0 from NVIDIA..."
    wget -q https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl -O torch-2.4.0-cp310-cp310-linux_aarch64.whl
fi

# Install NVIDIA wheel
echo "Installing NVIDIA PyTorch 2.4.0 (CUDA-enabled)..."
pip install --no-cache-dir --force-reinstall torch-2.4.0-cp310-cp310-linux_aarch64.whl

# Install torchvision/torchaudio with --no-deps
pip install --no-cache-dir torchvision==0.19.0 torchaudio==2.4.0 --no-deps
pip install --no-cache-dir pillow

# Apply cuDNN fixes
echo "Applying cuDNN compatibility fixes..."
./FIX_CUDNN_FOR_PYTORCH.sh 2>/dev/null || ./QUICK_FIX_ALL_LIBRARIES.sh

# Verify
echo ""
echo "Verifying installation..."
CUDA_COMPILED=$(python -c "import torch; import torch._C; print(hasattr(torch._C, '_cuda'))" 2>/dev/null || echo "False")
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

if [ "$CUDA_COMPILED" = "True" ]; then
    echo "✓ PyTorch compiled with CUDA support"
    if [ "$CUDA_AVAILABLE" = "True" ]; then
        echo "✓ CUDA is available!"
        python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU Count: {torch.cuda.device_count()}')"
    else
        echo "⚠ CUDA compiled but not available - check library fixes"
    fi
else
    echo "✗ PyTorch still CPU-only - installation may have failed"
    exit 1
fi
