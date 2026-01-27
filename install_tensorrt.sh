#!/bin/bash
# Install TensorRT for virtual environment on Jetson Nano
# TensorRT is already installed on the system, we just need to make it accessible

cd /home/ailab/Desktop/innovation_research_project

# Activate virtual environment
source .venv/bin/activate

# Create symlinks to system TensorRT installation
VENV_SITE_PACKAGES=".venv/lib/python3.10/site-packages"
SYSTEM_DIST_PACKAGES="/usr/lib/python3.10/dist-packages"

echo "Creating symlinks to system TensorRT installation..."

# Create symlinks for TensorRT modules
ln -sf "$SYSTEM_DIST_PACKAGES/tensorrt" "$VENV_SITE_PACKAGES/tensorrt"
ln -sf "$SYSTEM_DIST_PACKAGES/tensorrt_dispatch" "$VENV_SITE_PACKAGES/tensorrt_dispatch"
ln -sf "$SYSTEM_DIST_PACKAGES/tensorrt_lean" "$VENV_SITE_PACKAGES/tensorrt_lean"

# Verify installation
echo ""
echo "Testing TensorRT import..."
python3 -c "import tensorrt; print('✓ TensorRT version:', tensorrt.__version__)" && \
python3 -c "import pycuda.driver as cuda; import pycuda.autoinit; print('✓ PyCUDA OK')" && \
echo "" && \
echo "✓ TensorRT and PyCUDA are ready to use!"
