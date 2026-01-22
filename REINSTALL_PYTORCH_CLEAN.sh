#!/bin/bash
# ============================================================================
# Clean Reinstall PyTorch for Python 3.10 on Jetson Nano
# This fixes the "torch/_C folder" error caused by wrong Python version
# ============================================================================

set -e

cd /home/ailab/Desktop/innovation_research_project
source .venv/bin/activate

echo "Step 1: Uninstalling old PyTorch (wrong version)..."
pip uninstall -y torch torchvision torchaudio

echo ""
echo "Step 2: Verifying Python version (should be 3.10)..."
python --version

echo ""
echo "Step 3: Installing PyTorch 2.3.0 for Python 3.10..."
# Download PyTorch 2.3.0 wheel for Python 3.10 (cp310) from NVIDIA Box
if [ ! -f "torch-2.3.0-cp310-cp310-linux_aarch64.whl" ]; then
    echo "Downloading PyTorch 2.3.0 wheel from NVIDIA Box..."
    wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-2.3.0-cp310-cp310-linux_aarch64.whl
fi

echo "Installing PyTorch..."
pip install --no-cache-dir torch-2.3.0-cp310-cp310-linux_aarch64.whl

echo ""
echo "Step 4: Installing compatible numpy..."
pip install --no-cache-dir "numpy<2.0"

echo ""
echo "Step 5: Installing TorchVision and TorchAudio..."
# CRITICAL: Use --no-deps to prevent pip from installing CPU-only PyTorch from PyPI
echo "Installing with --no-deps to prevent dependency conflicts..."
pip install --no-cache-dir torchvision==0.18.0 torchaudio==2.3.0 --no-deps || echo "Note: torchvision/torchaudio may need manual installation"
pip install --no-cache-dir pillow  # Required dependency for torchvision

echo ""
echo "Step 6: Applying library fixes..."
cd /home/ailab/Desktop/innovation_research_project
./QUICK_FIX_ALL_LIBRARIES.sh
source ~/.bashrc

echo ""
echo "Step 7: Testing PyTorch..."
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA Available: {torch.cuda.is_available()}')"

echo ""
echo "Done! If you see errors, check the library fixes were applied."
