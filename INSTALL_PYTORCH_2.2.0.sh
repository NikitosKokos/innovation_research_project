#!/bin/bash
# ============================================================================
# Install PyTorch 2.2.0 - Better cuDNN 9 compatibility
# ============================================================================

set -e

cd /home/ailab/Desktop/innovation_research_project
source .venv/bin/activate

echo "Step 1: Uninstalling PyTorch 2.4.0..."
pip uninstall -y torch torchvision torchaudio

echo ""
echo "Step 2: Downloading PyTorch 2.2.0 (better cuDNN 9 compatibility)..."
wget https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.2.0a0+6a974be.nv23.11-cp310-cp310-linux_aarch64.whl -O torch-2.2.0-cp310-cp310-linux_aarch64.whl

echo ""
echo "Step 3: Installing PyTorch 2.2.0..."
pip install --no-cache-dir torch-2.2.0-cp310-cp310-linux_aarch64.whl

echo ""
echo "Step 4: Verifying installation..."
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" || echo "⚠ Import failed - will apply fixes"

echo ""
echo "Step 5: Applying cuDNN and library fixes..."
./FIX_CUDNN_FOR_PYTORCH.sh
./QUICK_FIX_ALL_LIBRARIES.sh || echo "⚠ Some fixes need sudo - run manually"
source ~/.bashrc

echo ""
echo "Step 6: Installing TorchVision and TorchAudio..."
pip install --no-cache-dir torchvision==0.17.0 torchaudio==2.2.0 --no-deps
pip install --no-cache-dir pillow

echo ""
echo "Step 7: Final verification..."
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA Available: {torch.cuda.is_available()}'); print(f'✓ CUDA Device Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"
