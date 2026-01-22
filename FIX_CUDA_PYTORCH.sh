#!/bin/bash
# ============================================================================
# Fix PyTorch Installation - Replace CPU-only with CUDA-enabled PyTorch
# ============================================================================

set -e

cd /home/ailab/Desktop/innovation_research_project
source .venv/bin/activate

echo "Step 1: Uninstalling CPU-only PyTorch..."
pip uninstall -y torch torchvision torchaudio

echo ""
echo "Step 2: Installing NVIDIA PyTorch 2.4.0 with CUDA support..."
# Use official NVIDIA redist URL (from requirements.txt)
echo "Downloading PyTorch 2.4.0 wheel from NVIDIA (official redist)..."
wget https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl -O torch-2.4.0-cp310-cp310-linux_aarch64.whl

echo "Installing PyTorch 2.4.0 (CUDA-enabled)..."
pip install --no-cache-dir torch-2.4.0-cp310-cp310-linux_aarch64.whl

echo ""
echo "Step 3: Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "✓ CUDA is available!"
else
    echo "✗ CUDA is NOT available - PyTorch may be CPU-only version"
    echo "  Checking installed PyTorch location..."
    pip show torch | grep Location
    exit 1
fi

echo ""
echo "Step 4: Installing TorchVision and TorchAudio with --no-deps..."
# CRITICAL: Use --no-deps to prevent pip from pulling in CPU-only PyTorch
pip install --no-cache-dir torchvision==0.19.0 torchaudio==2.4.0 --no-deps
pip install --no-cache-dir pillow

echo ""
echo "Step 5: Applying library fixes..."
./QUICK_FIX_ALL_LIBRARIES.sh
source ~/.bashrc

echo ""
echo "Step 6: Final verification..."
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA Available: {torch.cuda.is_available()}'); print(f'✓ CUDA Device Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

echo ""
echo "Done! PyTorch with CUDA support should now be working."
