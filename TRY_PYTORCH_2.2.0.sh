#!/bin/bash
# ============================================================================
# Try PyTorch 2.2.0 - May have better cuDNN 9 compatibility
# ============================================================================

set -e

cd /home/ailab/Desktop/innovation_research_project
source .venv/bin/activate

echo "Step 1: Uninstalling PyTorch 2.4.0..."
pip uninstall -y torch torchvision torchaudio

echo ""
echo "Step 2: Downloading PyTorch 2.2.0..."
# Try to find a 2.2.0 wheel - check NVIDIA forums for exact URL
echo "Searching for PyTorch 2.2.0 wheel..."
echo "Note: You may need to find the exact URL from NVIDIA forums"

# Common pattern for 2.2.0 wheels
# wget https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.2.0a0+[hash]-cp310-cp310-linux_aarch64.whl

echo ""
echo "Alternative: Try listing available wheels:"
echo "curl -s https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/ | grep 'torch-2.2'"

echo ""
echo "Or try PyTorch 2.3.0 from a different source..."
echo "The issue is that PyTorch 2.4.0 was compiled against cuDNN 8 symbols"
echo "and cannot work with cuDNN 9 without those symbols."
