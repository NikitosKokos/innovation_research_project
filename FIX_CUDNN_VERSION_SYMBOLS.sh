#!/bin/bash
# ============================================================================
# Fix cuDNN Version Symbols - More Aggressive Approach
# PyTorch binaries need libcudnn.so.8 version symbols, but system has cuDNN 9
# ============================================================================

set -e

VENV_TORCH_LIB="/home/ailab/Desktop/innovation_research_project/.venv/lib/python3.10/site-packages/torch/lib"
SYSTEM_CUDNN="/usr/lib/aarch64-linux-gnu/libcudnn.so.9"

echo "Fixing cuDNN version symbols for PyTorch..."

# Step 1: Ensure symlink exists
if [ -f "$SYSTEM_CUDNN" ]; then
    ln -sf "$SYSTEM_CUDNN" "$VENV_TORCH_LIB/libcudnn.so.8"
    echo "✓ Symlink created"
fi

# Step 2: Patch ALL binaries that reference libcudnn.so.8
if command -v patchelf >/dev/null 2>&1; then
    echo ""
    echo "Patching all PyTorch binaries..."
    
    for lib in "$VENV_TORCH_LIB"/*.so; do
        if [ -f "$lib" ]; then
            # Check if this library needs libcudnn.so.8
            if readelf -d "$lib" 2>/dev/null | grep -q "libcudnn.so.8"; then
                echo "  Patching: $(basename $lib)"
                patchelf --replace-needed libcudnn.so.8 libcudnn.so.9 "$lib" 2>/dev/null || true
            fi
        fi
    done
    echo "✓ All binaries patched"
else
    echo "✗ Error: patchelf not found. Install with: sudo apt-get install patchelf"
    exit 1
fi

# Step 3: Verify the patch
echo ""
echo "Verifying patches..."
if readelf -d "$VENV_TORCH_LIB/libtorch_python.so" 2>/dev/null | grep -q "libcudnn.so.9"; then
    echo "✓ libtorch_python.so now uses libcudnn.so.9"
else
    echo "⚠ Warning: libtorch_python.so may still reference libcudnn.so.8"
fi

echo ""
echo "Note: If you still get version symbol errors, you may need to:"
echo "  1. Use LD_PRELOAD to force library loading"
echo "  2. Or use a Docker container with compatible cuDNN 8"
echo "  3. Or try PyTorch 2.2.0/2.3.0 which may have better cuDNN 9 compatibility"
