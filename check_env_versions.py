import os
import sys
import torch
import numpy as np

def check_environment():
    print("=== System Environment Check ===")
    
    # Check PyTorch
    print(f"\n[PyTorch]")
    print(f"Version: {torch.__version__}")
    try:
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Device Name: {torch.cuda.get_device_name(0)}")
            print(f"Device Capability: {torch.cuda.get_device_capability(0)}")
            print(f"CuDNN Version: {torch.backends.cudnn.version()}")
            print(f"CuDNN Benchmark: {torch.backends.cudnn.benchmark}")
            print(f"FP16 Support: {True}") # Jetson supports FP16
    except Exception as e:
        print(f"Error checking CUDA: {e}")

    # Check NumPy
    print(f"\n[NumPy]")
    print(f"Version: {np.__version__}")

    # Check TensorRT (if available)
    print(f"\n[TensorRT]")
    try:
        import tensorrt as trt
        print(f"Version: {trt.__version__}")
    except ImportError:
        print("TensorRT python bindings not installed (normal if using PyTorch only)")

if __name__ == "__main__":
    check_environment()
