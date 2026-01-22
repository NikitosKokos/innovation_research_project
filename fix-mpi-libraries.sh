#!/bin/bash
# ============================================================================
# Fix MPI Library Compatibility for PyTorch on Jetson Nano
# PyTorch expects MPI 2.x (libmpi.so.20) but system has OpenMPI 4.x (libmpi.so.40)
# ============================================================================

echo "Fixing MPI library compatibility..."

# Create symlinks for both libmpi.so.20 and libmpi_cxx.so.20
sudo ln -sf /usr/lib/aarch64-linux-gnu/libmpi.so.40.30.2 /usr/lib/aarch64-linux-gnu/libmpi.so.20
sudo ln -sf /usr/lib/aarch64-linux-gnu/libmpi_cxx.so.40 /usr/lib/aarch64-linux-gnu/libmpi_cxx.so.20

# Update library cache
sudo ldconfig

echo "✓ MPI compatibility fixes applied"
echo ""
echo "Verify with:"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}\")'"
