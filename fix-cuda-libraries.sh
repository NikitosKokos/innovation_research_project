#!/bin/bash
# ============================================================================
# Fix CUDA Library Compatibility for PyTorch on Jetson Nano
# PyTorch expects CUDA 10.x libraries but system has CUDA 12.6
# ============================================================================

echo "Fixing CUDA library compatibility..."

CUDA_LIB_DIR="/usr/local/cuda-12.6/targets/aarch64-linux/lib"
SYSTEM_LIB_DIR="/usr/lib/aarch64-linux-gnu"

# Fix libcufft.so.10 -> libcufft.so.11
if [ -f "$CUDA_LIB_DIR/libcufft.so.11" ]; then
    sudo ln -sf "$CUDA_LIB_DIR/libcufft.so.11" "$CUDA_LIB_DIR/libcufft.so.10"
    sudo ln -sf "$CUDA_LIB_DIR/libcufft.so.11" "$SYSTEM_LIB_DIR/libcufft.so.10"
    echo "✓ Fixed libcufft.so.10"
fi

# Fix libcublas.so.10 -> libcublas.so.12 (or latest available)
if [ -f "$CUDA_LIB_DIR/libcublas.so.12" ]; then
    sudo ln -sf "$CUDA_LIB_DIR/libcublas.so.12" "$CUDA_LIB_DIR/libcublas.so.10"
    sudo ln -sf "$CUDA_LIB_DIR/libcublas.so.12" "$SYSTEM_LIB_DIR/libcublas.so.10"
    echo "✓ Fixed libcublas.so.10"
elif [ -f "$CUDA_LIB_DIR/libcublas.so" ]; then
    # Try to find the actual versioned library
    CUBLAS_LIB=$(ls "$CUDA_LIB_DIR"/libcublas.so.* 2>/dev/null | head -1)
    if [ -n "$CUBLAS_LIB" ]; then
        sudo ln -sf "$CUBLAS_LIB" "$CUDA_LIB_DIR/libcublas.so.10"
        sudo ln -sf "$CUBLAS_LIB" "$SYSTEM_LIB_DIR/libcublas.so.10"
        echo "✓ Fixed libcublas.so.10"
    fi
fi

# Update library cache
sudo ldconfig

echo ""
echo "✓ CUDA compatibility fixes applied"
echo ""
echo "Note: You may also need to set LD_LIBRARY_PATH in your ~/.bashrc:"
echo "  export LD_LIBRARY_PATH=$CUDA_LIB_DIR:\$LD_LIBRARY_PATH"
