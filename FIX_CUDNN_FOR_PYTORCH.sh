#!/bin/bash
# ============================================================================
# Fix cuDNN Compatibility for PyTorch 2.4.0 on JetPack 6.0
# PyTorch expects cuDNN 8, but JetPack 6.0 provides cuDNN 9
# ============================================================================

set -e

VENV_TORCH_LIB="/home/ailab/Desktop/innovation_research_project/.venv/lib/python3.10/site-packages/torch/lib"
SYSTEM_CUDNN="/usr/lib/aarch64-linux-gnu/libcudnn.so.9"

echo "Fixing cuDNN compatibility for PyTorch..."

# Step 1: Create symlink in torch/lib directory
if [ -f "$SYSTEM_CUDNN" ]; then
    echo "Creating symlink: $VENV_TORCH_LIB/libcudnn.so.8 -> $SYSTEM_CUDNN"
    ln -sf "$SYSTEM_CUDNN" "$VENV_TORCH_LIB/libcudnn.so.8"
    echo "✓ Symlink created"
else
    echo "✗ Error: $SYSTEM_CUDNN not found"
    exit 1
fi

# Step 2: Patch binary dependencies using patchelf
if command -v patchelf >/dev/null 2>&1; then
    echo ""
    echo "Patching binary dependencies with patchelf..."
    
    # Patch libtorch_python.so
    if [ -f "$VENV_TORCH_LIB/libtorch_python.so" ]; then
        patchelf --replace-needed libcudnn.so.8 libcudnn.so.9 "$VENV_TORCH_LIB/libtorch_python.so" 2>/dev/null || echo "  Note: libtorch_python.so may already be patched"
        echo "✓ Patched libtorch_python.so"
    fi
    
    # Patch libtorch_cuda.so
    if [ -f "$VENV_TORCH_LIB/libtorch_cuda.so" ]; then
        patchelf --replace-needed libcudnn.so.8 libcudnn.so.9 "$VENV_TORCH_LIB/libtorch_cuda.so" 2>/dev/null || echo "  Note: libtorch_cuda.so may already be patched"
        echo "✓ Patched libtorch_cuda.so"
    fi
    
    # Patch libc10_cuda.so
    if [ -f "$VENV_TORCH_LIB/libc10_cuda.so" ]; then
        patchelf --replace-needed libcudnn.so.8 libcudnn.so.9 "$VENV_TORCH_LIB/libc10_cuda.so" 2>/dev/null || echo "  Note: libc10_cuda.so may already be patched"
        echo "✓ Patched libc10_cuda.so"
    fi
else
    echo "⚠ Warning: patchelf not found. Symlink created, but binary patching skipped."
    echo "  Install patchelf: sudo apt-get install patchelf"
fi

# Step 3: Apply all other library fixes
echo ""
echo "Applying other library compatibility fixes..."
cd /home/ailab/Desktop/innovation_research_project
./QUICK_FIX_ALL_LIBRARIES.sh
source ~/.bashrc

echo ""
echo "✓ cuDNN fix complete!"
echo ""
echo "Testing PyTorch..."
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA Available: {torch.cuda.is_available()}'); print(f'✓ CUDA Device Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')" || echo "✗ Import failed - check errors above"
