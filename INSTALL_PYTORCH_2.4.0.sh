#!/bin/bash
# ============================================================================
# Install PyTorch 2.4.0 for JetPack 6.0 (Official NVIDIA Wheel)
# This is the version documented in requirements.txt
# ============================================================================

set -e

cd /home/ailab/Desktop/innovation_research_project
source .venv/bin/activate

echo "Step 1: Uninstalling any existing PyTorch..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

echo ""
echo "Step 2: Downloading PyTorch 2.4.0 from NVIDIA (official redist)..."
# Official URL from requirements.txt
wget https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl -O torch-2.4.0-cp310-cp310-linux_aarch64.whl

echo ""
echo "Step 3: Installing PyTorch 2.4.0..."
pip install --no-cache-dir torch-2.4.0-cp310-cp310-linux_aarch64.whl

echo ""
echo "Step 4: Verifying installation..."
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "⚠ CUDA not available - will need library fixes"
fi

echo ""
echo "Step 5: Installing TorchVision and TorchAudio with --no-deps..."
pip install --no-cache-dir torchvision==0.19.0 torchaudio==2.4.0 --no-deps
pip install --no-cache-dir pillow

echo ""
echo "Step 6: Applying cuDNN and library compatibility fixes..."
# First fix cuDNN specifically (PyTorch 2.4.0 needs this)
./FIX_CUDNN_FOR_PYTORCH.sh
source ~/.bashrc

echo ""
echo "Step 7: Final verification..."
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA Available: {torch.cuda.is_available()}'); print(f'✓ CUDA Device Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

echo ""
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "✅ SUCCESS! PyTorch with CUDA support is working!"
else
    echo "⚠ PyTorch installed but CUDA not available - check library fixes"
fi
