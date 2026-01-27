#!/bin/bash
# ============================================================================
# COMPLETE SETUP SCRIPT FOR JETSON NANO (JetPack 6.0)
# Based on official NVIDIA documentation and best practices
# This script installs everything in the correct order
# ============================================================================

set -e  # Exit on error

PROJECT_DIR="/home/ailab/Desktop/innovation_research_project"
cd "$PROJECT_DIR"

echo "============================================================================"
echo "COMPLETE JETSON NANO SETUP - JetPack 6.0"
echo "============================================================================"
echo ""

# ============================================================================
# STEP 1: Clean up any existing broken installations
# ============================================================================
echo "[1/8] Cleaning up existing installations..."
if [ -d ".venv" ]; then
    echo "  Removing existing virtual environment..."
    rm -rf .venv
fi

# Remove any existing PyTorch wheels
rm -f torch-*.whl

echo "✓ Cleanup complete"
echo ""

# ============================================================================
# STEP 2: Install system dependencies
# ============================================================================
echo "[2/8] Installing system dependencies..."
echo "  (This requires sudo - you'll be prompted for password)"

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

echo "✓ System dependencies installed"
echo ""

# ============================================================================
# STEP 3: Create fresh virtual environment
# ============================================================================
echo "[3/8] Creating fresh virtual environment (Python 3.10)..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

echo "✓ Virtual environment created"
echo ""

# ============================================================================
# STEP 4: Install NumPy first (required by PyTorch)
# ============================================================================
echo "[4/8] Installing NumPy (required by PyTorch)..."
pip install --no-cache-dir "numpy<2.0"

echo "✓ NumPy installed"
echo ""

# ============================================================================
# STEP 5: Install PyTorch from Jetson AI Lab repository (cuDNN 9 compatible)
# ============================================================================
echo "[5/8] Installing PyTorch with CUDA support..."
echo "  Using Jetson AI Lab repository (better cuDNN 9 compatibility)..."

# Method 1: Try PyTorch 2.2.0 first (better cuDNN 9 compatibility)
echo "  Trying PyTorch 2.2.0 (better cuDNN 9 support)..."
wget -q https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.2.0a0+6a974be.nv23.11-cp310-cp310-linux_aarch64.whl -O torch-2.2.0-cp310-cp310-linux_aarch64.whl || {
    echo "  Primary URL failed, trying alternative..."
    wget -q https://developer.download.nvidia.cn/compute/redist/jp/v60dp/pytorch/torch-2.2.0a0+6a974be.nv23.11-cp310-cp310-linux_aarch64.whl -O torch-2.2.0-cp310-cp310-linux_aarch64.whl
}

if [ -f "torch-2.2.0-cp310-cp310-linux_aarch64.whl" ]; then
    pip install --no-cache-dir torch-2.2.0-cp310-cp310-linux_aarch64.whl
    pip install --no-cache-dir torchvision==0.17.0 torchaudio==2.2.0 --no-deps
    pip install --no-cache-dir pillow
    PYTORCH_INSTALLED=true
else
    PYTORCH_INSTALLED=false
fi

# If 2.2.0 fails, try 2.4.0 as fallback
if [ "$PYTORCH_INSTALLED" != "true" ] || ! python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    echo "  PyTorch 2.2.0 failed or CUDA unavailable, trying PyTorch 2.4.0..."
    pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    
    # Download and install PyTorch 2.4.0 from official NVIDIA redist
    echo "  Downloading PyTorch 2.4.0 from NVIDIA..."
    if [ ! -f "torch-2.4.0-cp310-cp310-linux_aarch64.whl" ]; then
        wget -q https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl -O torch-2.4.0-cp310-cp310-linux_aarch64.whl || {
            echo "  Primary URL failed, trying alternative..."
            wget -q https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl -O torch-2.4.0-cp310-cp310-linux_aarch64.whl
        }
    fi
    
    echo "  Installing PyTorch 2.4.0 from NVIDIA wheel..."
    pip install --no-cache-dir --force-reinstall torch-2.4.0-cp310-cp310-linux_aarch64.whl
    
    # Install torchvision and torchaudio with --no-deps to prevent conflicts
    pip install --no-cache-dir torchvision==0.19.0 torchaudio==2.4.0 --no-deps
    pip install --no-cache-dir pillow  # Required by torchvision
    
    PYTORCH_INSTALLED=true
fi

echo "✓ PyTorch installed"
echo ""

# ============================================================================
# STEP 6: Apply library compatibility fixes
# ============================================================================
echo "[6/8] Applying library compatibility fixes..."

# Fix cuDNN (create symlink in torch/lib)
VENV_TORCH_LIB="$PROJECT_DIR/.venv/lib/python3.10/site-packages/torch/lib"
SYSTEM_CUDNN="/usr/lib/aarch64-linux-gnu/libcudnn.so.9"

if [ -f "$SYSTEM_CUDNN" ] && [ -d "$VENV_TORCH_LIB" ]; then
    ln -sf "$SYSTEM_CUDNN" "$VENV_TORCH_LIB/libcudnn.so.8" 2>/dev/null || true
    echo "  ✓ Created cuDNN symlink"
    
    # Patch binaries with patchelf
    if command -v patchelf >/dev/null 2>&1; then
        for lib in "$VENV_TORCH_LIB"/*.so; do
            if [ -f "$lib" ] && readelf -d "$lib" 2>/dev/null | grep -q "libcudnn.so.8"; then
                patchelf --replace-needed libcudnn.so.8 libcudnn.so.9 "$lib" 2>/dev/null || true
            fi
        done
        echo "  ✓ Patched PyTorch binaries"
    fi
fi

# Fix system libraries (requires sudo)
echo "  Applying system library fixes (requires sudo)..."
sudo bash << 'EOF'
CUDA_LIB_DIR="/usr/local/cuda-12.6/targets/aarch64-linux/lib"
SYSTEM_LIB_DIR="/usr/lib/aarch64-linux-gnu"

# cuDNN
ln -sf "$SYSTEM_LIB_DIR/libcudnn.so.9" "$SYSTEM_LIB_DIR/libcudnn.so.8" 2>/dev/null || true

# MPI
ln -sf "$SYSTEM_LIB_DIR/libmpi.so.40.30.2" "$SYSTEM_LIB_DIR/libmpi.so.20" 2>/dev/null || true
ln -sf "$SYSTEM_LIB_DIR/libmpi_cxx.so.40" "$SYSTEM_LIB_DIR/libmpi_cxx.so.20" 2>/dev/null || true

# CUDA Libraries
ln -sf "$CUDA_LIB_DIR/libcufft.so.11" "$CUDA_LIB_DIR/libcufft.so.10" 2>/dev/null || true
ln -sf "$CUDA_LIB_DIR/libcufft.so.11" "$SYSTEM_LIB_DIR/libcufft.so.10" 2>/dev/null || true

ln -sf "$CUDA_LIB_DIR/libcublas.so.12" "$CUDA_LIB_DIR/libcublas.so.10" 2>/dev/null || true
ln -sf "$CUDA_LIB_DIR/libcublas.so.12" "$SYSTEM_LIB_DIR/libcublas.so.10" 2>/dev/null || true

ln -sf "$CUDA_LIB_DIR/libcudart.so.12" "$CUDA_LIB_DIR/libcudart.so.10" 2>/dev/null || true
ln -sf "$CUDA_LIB_DIR/libcudart.so.12" "$CUDA_LIB_DIR/libcudart.so.10.2" 2>/dev/null || true
ln -sf "$CUDA_LIB_DIR/libcudart.so.12" "$SYSTEM_LIB_DIR/libcudart.so.10" 2>/dev/null || true
ln -sf "$CUDA_LIB_DIR/libcudart.so.12" "$SYSTEM_LIB_DIR/libcudart.so.10.2" 2>/dev/null || true

ldconfig
EOF

# Set LD_LIBRARY_PATH
if ! grep -q "cuda-12.6/targets/aarch64-linux/lib" ~/.bashrc 2>/dev/null; then
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/aarch64-linux/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
fi
source ~/.bashrc

echo "✓ Library fixes applied"
echo ""

# ============================================================================
# STEP 7: Install remaining Python packages (excluding PyTorch)
# ============================================================================
echo "[7/8] Installing remaining Python packages..."
echo "  Note: PyTorch is already installed, preventing overwrite..."

# Create temporary requirements file without PyTorch-related packages
grep -v -E "^(torch|torchvision|torchaudio)" requirements-jetson-nano.txt > /tmp/requirements_no_torch.txt 2>/dev/null || cp requirements-jetson-nano.txt /tmp/requirements_no_torch.txt

# Install packages, explicitly excluding PyTorch to prevent overwrite
pip install --no-cache-dir -r /tmp/requirements_no_torch.txt 2>/dev/null || {
    echo "  Installing with dependency resolution disabled to protect PyTorch..."
    pip install --no-cache-dir -r requirements-jetson-nano.txt --no-deps 2>/dev/null || true
}

# Verify PyTorch wasn't overwritten with CPU-only version
CUDA_COMPILED=$(python -c "import torch; import torch._C; print(hasattr(torch._C, '_cuda'))" 2>/dev/null || echo "False")
if [ "$CUDA_COMPILED" = "False" ]; then
    echo "  ⚠ PyTorch was overwritten with CPU-only version. Fixing..."
    if [ -f "torch-2.4.0-cp310-cp310-linux_aarch64.whl" ]; then
        pip install --no-cache-dir --force-reinstall torch-2.4.0-cp310-cp310-linux_aarch64.whl
        pip install --no-cache-dir torchvision==0.19.0 torchaudio==2.4.0 --no-deps
        echo "  ✓ PyTorch NVIDIA wheel reinstalled"
    else
        echo "  ⚠ NVIDIA wheel not found. Run ./FIX_PYTORCH_CUDA.sh to fix"
    fi
else
    echo "  ✓ PyTorch NVIDIA wheel preserved"
fi

rm -f /tmp/requirements_no_torch.txt

echo "✓ Python packages installed"
echo ""

# ============================================================================
# STEP 8: Verify installation
# ============================================================================
echo "[8/8] Verifying installation..."
source .venv/bin/activate

python << 'PYTHON_VERIFY'
import sys
import os

def diagnose_cuda_issue():
    """Provide detailed diagnostics for CUDA availability issues"""
    issues = []
    
    # Check PyTorch version
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    
    # Check if CUDA is compiled in
    try:
        has_cuda = hasattr(torch._C, '_cuda')
        if not has_cuda:
            issues.append("PyTorch was compiled without CUDA support (CPU-only version)")
    except:
        issues.append("Cannot check if PyTorch has CUDA support")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        issues.append("torch.cuda.is_available() returned False")
        
        # Check for common issues
        if "cpu" in torch.__version__.lower():
            issues.append("PyTorch version indicates CPU-only build")
        
        # Check environment variables
        cuda_home = os.environ.get('CUDA_HOME', '')
        ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        if 'cuda' not in ld_path.lower():
            issues.append("LD_LIBRARY_PATH may not include CUDA libraries")
    
    return issues, cuda_available

try:
    issues, cuda_available = diagnose_cuda_issue()
    
    if cuda_available:
        import torch
        print(f"✓ CUDA available: True")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU count: {torch.cuda.device_count()}")
        print(f"✓ GPU name: {torch.cuda.get_device_name(0)}")
        
        # Test basic operations
        x = torch.randn(3, 3).cuda()
        y = torch.randn(3, 3).cuda()
        z = x @ y
        print("✓ GPU computation test: PASSED")
        
        print("\n✅ ALL CHECKS PASSED - PyTorch is working correctly!")
    else:
        print("\n⚠ CUDA NOT AVAILABLE - Diagnostic Information:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        print("\nPossible solutions:")
        print("  1. Reinstall PyTorch from NVIDIA wheel (not PyPI)")
        print("  2. Apply cuDNN compatibility fixes")
        print("  3. Check LD_LIBRARY_PATH includes CUDA libraries")
        print("  4. Verify CUDA installation: nvidia-smi")
        
        sys.exit(1)
    
except Exception as e:
    print(f"\n✗ VERIFICATION FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_VERIFY

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================================"
    echo "✅ SETUP COMPLETE - PyTorch is working with CUDA!"
    echo "============================================================================"
    echo ""
    echo "To activate the environment:"
    echo "  cd $PROJECT_DIR"
    echo "  source .venv/bin/activate"
    echo ""
else
    echo ""
    echo "============================================================================"
    echo "⚠ SETUP COMPLETED WITH WARNINGS"
    echo "============================================================================"
    echo "PyTorch installed but CUDA verification failed."
    echo ""
    echo "Next steps to fix CUDA availability:"
    echo "  1. Re-run cuDNN fixes: ./FIX_CUDNN_FOR_PYTORCH.sh"
    echo "  2. Verify PyTorch is NVIDIA wheel (not CPU-only):"
    echo "     pip show torch | grep Version"
    echo "  3. Check CUDA installation: nvidia-smi"
    echo "  4. Verify library paths: echo \$LD_LIBRARY_PATH"
    echo ""
    echo "If issues persist, try PyTorch 2.2.0 instead:"
    echo "  ./INSTALL_PYTORCH_2.2.0.sh"
    echo ""
fi
