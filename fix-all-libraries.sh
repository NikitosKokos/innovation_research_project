#!/bin/bash
# ============================================================================
# Complete Library Compatibility Fix for PyTorch on Jetson Nano (JetPack 6.0)
# Fixes: cuDNN, MPI, and CUDA library version mismatches
# ============================================================================

set -e

echo "=========================================="
echo "Fixing All Library Compatibility Issues"
echo "=========================================="

# CUDA Library Directory
CUDA_LIB_DIR="/usr/local/cuda-12.6/targets/aarch64-linux/lib"
SYSTEM_LIB_DIR="/usr/lib/aarch64-linux-gnu"

# 1. Fix cuDNN (libcudnn.so.8 -> libcudnn.so.9)
echo ""
echo "[1/4] Fixing cuDNN compatibility..."
if [ -f "$SYSTEM_LIB_DIR/libcudnn.so.9" ]; then
    sudo ln -sf "$SYSTEM_LIB_DIR/libcudnn.so.9" "$SYSTEM_LIB_DIR/libcudnn.so.8"
    echo "  ✓ Created libcudnn.so.8 symlink"
else
    echo "  ⚠ Warning: libcudnn.so.9 not found"
fi

# 2. Fix MPI (libmpi.so.20 -> libmpi.so.40)
echo ""
echo "[2/4] Fixing MPI compatibility..."
if [ -f "$SYSTEM_LIB_DIR/libmpi.so.40.30.2" ]; then
    sudo ln -sf "$SYSTEM_LIB_DIR/libmpi.so.40.30.2" "$SYSTEM_LIB_DIR/libmpi.so.20"
    echo "  ✓ Created libmpi.so.20 symlink"
fi
if [ -f "$SYSTEM_LIB_DIR/libmpi_cxx.so.40" ]; then
    sudo ln -sf "$SYSTEM_LIB_DIR/libmpi_cxx.so.40" "$SYSTEM_LIB_DIR/libmpi_cxx.so.20"
    echo "  ✓ Created libmpi_cxx.so.20 symlink"
fi

# 3. Fix CUDA libraries (libcufft.so.10, libcublas.so.10, libcudart.so.10, etc.)
echo ""
echo "[3/4] Fixing CUDA library compatibility..."
# Fix libcufft.so.10
if [ -f "$CUDA_LIB_DIR/libcufft.so.11" ]; then
    sudo ln -sf "$CUDA_LIB_DIR/libcufft.so.11" "$CUDA_LIB_DIR/libcufft.so.10"
    sudo ln -sf "$CUDA_LIB_DIR/libcufft.so.11" "$SYSTEM_LIB_DIR/libcufft.so.10"
    echo "  ✓ Created libcufft.so.10 symlink"
fi
# Fix libcublas.so.10
if [ -f "$CUDA_LIB_DIR/libcublas.so.12" ]; then
    sudo ln -sf "$CUDA_LIB_DIR/libcublas.so.12" "$CUDA_LIB_DIR/libcublas.so.10"
    sudo ln -sf "$CUDA_LIB_DIR/libcublas.so.12" "$SYSTEM_LIB_DIR/libcublas.so.10"
    echo "  ✓ Created libcublas.so.10 symlink"
fi
# Fix libcudart.so.10.2 (CUDA Runtime - PyTorch needs specific version 10.2)
if [ -f "$CUDA_LIB_DIR/libcudart.so.12" ]; then
    # Create both libcudart.so.10 and libcudart.so.10.2
    sudo ln -sf "$CUDA_LIB_DIR/libcudart.so.12" "$CUDA_LIB_DIR/libcudart.so.10"
    sudo ln -sf "$CUDA_LIB_DIR/libcudart.so.12" "$CUDA_LIB_DIR/libcudart.so.10.2"
    sudo ln -sf "$CUDA_LIB_DIR/libcudart.so.12" "$SYSTEM_LIB_DIR/libcudart.so.10"
    sudo ln -sf "$CUDA_LIB_DIR/libcudart.so.12" "$SYSTEM_LIB_DIR/libcudart.so.10.2"
    echo "  ✓ Created libcudart.so.10 and libcudart.so.10.2 symlinks"
elif [ -f "$CUDA_LIB_DIR/libcudart.so" ]; then
    CUDART_LIB=$(ls "$CUDA_LIB_DIR"/libcudart.so.* 2>/dev/null | grep -v ".so$" | head -1)
    if [ -n "$CUDART_LIB" ]; then
        sudo ln -sf "$CUDART_LIB" "$CUDA_LIB_DIR/libcudart.so.10"
        sudo ln -sf "$CUDART_LIB" "$CUDA_LIB_DIR/libcudart.so.10.2"
        sudo ln -sf "$CUDART_LIB" "$SYSTEM_LIB_DIR/libcudart.so.10"
        sudo ln -sf "$CUDART_LIB" "$SYSTEM_LIB_DIR/libcudart.so.10.2"
        echo "  ✓ Created libcudart.so.10 and libcudart.so.10.2 symlinks"
    fi
fi
# Fix other common CUDA libraries that might be needed
for lib in libcurand libcusolver libcusparse libcudnn; do
    if [ -f "$CUDA_LIB_DIR/${lib}.so.12" ] || [ -f "$CUDA_LIB_DIR/${lib}.so.11" ]; then
        LATEST=$(ls "$CUDA_LIB_DIR"/${lib}.so.* 2>/dev/null | grep -v ".so$" | sort -V | tail -1)
        if [ -n "$LATEST" ]; then
            sudo ln -sf "$LATEST" "$CUDA_LIB_DIR/${lib}.so.10" 2>/dev/null || true
            sudo ln -sf "$LATEST" "$SYSTEM_LIB_DIR/${lib}.so.10" 2>/dev/null || true
        fi
    fi
done

# 4. Update library cache
echo ""
echo "[4/4] Updating library cache..."
sudo ldconfig

# 5. Set LD_LIBRARY_PATH (add to .bashrc if not already there)
echo ""
echo "Setting up LD_LIBRARY_PATH..."
if ! grep -q "cuda-12.6/targets/aarch64-linux/lib" ~/.bashrc 2>/dev/null; then
    echo "export LD_LIBRARY_PATH=$CUDA_LIB_DIR:\$LD_LIBRARY_PATH" >> ~/.bashrc
    echo "  ✓ Added CUDA library path to ~/.bashrc"
else
    echo "  ✓ CUDA library path already in ~/.bashrc"
fi

echo ""
echo "=========================================="
echo "✓ All library fixes applied!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Reload your environment:"
echo "     source ~/.bashrc"
echo ""
echo "  2. Test PyTorch:"
echo "     source .venv/bin/activate"
echo "     python -c \"import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')\""
echo ""
