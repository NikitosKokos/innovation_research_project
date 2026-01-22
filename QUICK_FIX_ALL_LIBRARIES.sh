#!/bin/bash
# ============================================================================
# QUICK FIX: All Library Compatibility Issues for PyTorch on Jetson Nano
# Run this script to fix ALL library issues at once
# ============================================================================

set -e

CUDA_LIB_DIR="/usr/local/cuda-12.6/targets/aarch64-linux/lib"
SYSTEM_LIB_DIR="/usr/lib/aarch64-linux-gnu"

echo "Applying all library compatibility fixes..."

# 1. cuDNN
sudo ln -sf "$SYSTEM_LIB_DIR/libcudnn.so.9" "$SYSTEM_LIB_DIR/libcudnn.so.8" 2>/dev/null || true

# 2. MPI
sudo ln -sf "$SYSTEM_LIB_DIR/libmpi.so.40.30.2" "$SYSTEM_LIB_DIR/libmpi.so.20" 2>/dev/null || true
sudo ln -sf "$SYSTEM_LIB_DIR/libmpi_cxx.so.40" "$SYSTEM_LIB_DIR/libmpi_cxx.so.20" 2>/dev/null || true

# 3. CUDA Libraries
sudo ln -sf "$CUDA_LIB_DIR/libcufft.so.11" "$CUDA_LIB_DIR/libcufft.so.10" 2>/dev/null || true
sudo ln -sf "$CUDA_LIB_DIR/libcufft.so.11" "$SYSTEM_LIB_DIR/libcufft.so.10" 2>/dev/null || true

sudo ln -sf "$CUDA_LIB_DIR/libcublas.so.12" "$CUDA_LIB_DIR/libcublas.so.10" 2>/dev/null || true
sudo ln -sf "$CUDA_LIB_DIR/libcublas.so.12" "$SYSTEM_LIB_DIR/libcublas.so.10" 2>/dev/null || true

sudo ln -sf "$CUDA_LIB_DIR/libcudart.so.12" "$CUDA_LIB_DIR/libcudart.so.10" 2>/dev/null || true
sudo ln -sf "$CUDA_LIB_DIR/libcudart.so.12" "$CUDA_LIB_DIR/libcudart.so.10.2" 2>/dev/null || true
sudo ln -sf "$CUDA_LIB_DIR/libcudart.so.12" "$SYSTEM_LIB_DIR/libcudart.so.10" 2>/dev/null || true
sudo ln -sf "$CUDA_LIB_DIR/libcudart.so.12" "$SYSTEM_LIB_DIR/libcudart.so.10.2" 2>/dev/null || true

# Update cache
sudo ldconfig

# Set LD_LIBRARY_PATH
if ! grep -q "cuda-12.6/targets/aarch64-linux/lib" ~/.bashrc 2>/dev/null; then
    echo "export LD_LIBRARY_PATH=$CUDA_LIB_DIR:\$LD_LIBRARY_PATH" >> ~/.bashrc
fi

echo "✓ All fixes applied! Run: source ~/.bashrc"
