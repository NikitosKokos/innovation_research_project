#!/usr/bin/env python3
"""
Convert Seed-VC model to TensorRT engine for faster inference on Jetson Nano.
"""
import os
import sys
import argparse
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

def convert_to_tensorrt(checkpoint_path, config_path, output_path, precision='FP16'):
    """
    Convert PyTorch model to TensorRT engine.
    
    Args:
        checkpoint_path: Path to PyTorch checkpoint
        config_path: Path to model config YAML
        config_path: Path to output TensorRT engine
        precision: 'FP16' or 'FP32'
    """
    print(f"[TensorRT] Converting model to TensorRT {precision}...")
    print(f"[TensorRT] Checkpoint: {checkpoint_path}")
    print(f"[TensorRT] Config: {config_path}")
    print(f"[TensorRT] Output: {output_path}")
    
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError as e:
        print(f"[Error] TensorRT or PyCUDA not available: {e}")
        print("[Info] Install with: pip install nvidia-tensorrt pycuda")
        return False
    
    # First, convert to ONNX
    print("[TensorRT] Step 1: Converting to ONNX...")
    onnx_path = output_path.replace('.trt', '.onnx')
    
    # Use the existing convert_to_onnx script
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    onnx_converter = os.path.join(project_root, "project", "edge_optimization", "convert_to_onnx.py")
    
    if not os.path.exists(onnx_converter):
        print(f"[Error] ONNX converter not found: {onnx_converter}")
        return False
    
    import subprocess
    onnx_cmd = [
        sys.executable, onnx_converter,
        "--checkpoint", checkpoint_path,
        "--config", config_path,
        "--output", onnx_path
    ]
    
    print(f"[TensorRT] Running: {' '.join(onnx_cmd)}")
    result = subprocess.run(onnx_cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
    
    if result.returncode != 0:
        print(f"[Error] ONNX conversion failed:")
        print(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)  # Last 1000 chars
        print("\n[Info] ONNX export may not be supported for this model architecture.")
        print("       The Seed-VC DiT model has complex operations that ONNX can't handle.")
        print("\n[Alternative] Use the PyTorch optimizations instead:")
        print("  - FP16 quantization (enabled in config)")
        print("  - Reduced diffusion steps (3 steps)")
        print("  - CFG disabled (0.0)")
        print("  - Optimized block size")
        print("\nThese should provide good real-time performance on Jetson Nano.")
        return False
    
    if not os.path.exists(onnx_path):
        print(f"[Error] ONNX file not created: {onnx_path}")
        return False
    
    print(f"[TensorRT] ONNX conversion successful: {onnx_path}")
    
    # Now convert ONNX to TensorRT
    print("[TensorRT] Step 2: Converting ONNX to TensorRT engine...")
    
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("[Error] Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB workspace for Jetson
    
    if precision == 'FP16':
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[TensorRT] FP16 precision enabled")
        else:
            print("[Warning] FP16 not supported on this platform, using FP32")
    elif precision == 'INT8':
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("[Warning] INT8 requires calibration - not implemented yet")
        else:
            print("[Warning] INT8 not supported on this platform, using FP32")
    
    # Load config to get model dimensions
    import yaml
    with open(config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    model_params = model_config.get('model_params', {})
    dit_params = model_params.get('DiT', {})
    in_channels = dit_params.get('in_channels', 80)
    content_dim = dit_params.get('content_dim', 512)
    style_dim = model_params.get('style_encoder', {}).get('dim', 192)
    
    # Define optimization profiles for dynamic shapes
    # For real-time, we use smaller sequences
    T_min, T_opt, T_max = 32, 128, 256  # Optimized for real-time chunks
    
    profile = builder.create_optimization_profile()
    profile.set_shape("x", (1, in_channels, T_min), (1, in_channels, T_opt), (1, in_channels, T_max))
    profile.set_shape("prompt_x", (1, in_channels, T_min), (1, in_channels, T_opt), (1, in_channels, T_max))
    profile.set_shape("x_lens", (1,), (1,), (1,))
    profile.set_shape("t", (1,), (1,), (1,))
    profile.set_shape("style", (1, style_dim), (1, style_dim), (1, style_dim))
    profile.set_shape("cond", (1, T_min, content_dim), (1, T_opt, content_dim), (1, T_max, content_dim))
    config.add_optimization_profile(profile)
    
    print("[TensorRT] Building engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("[Error] Failed to build TensorRT engine")
        return False
    
    # Save engine
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(serialized_engine)
    
    engine_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[TensorRT] Engine saved: {output_path} ({engine_size_mb:.2f} MB)")
    
    # Clean up ONNX file (optional)
    # os.remove(onnx_path)
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Seed-VC model to TensorRT")
    parser.add_argument("--checkpoint", required=True, help="Path to PyTorch checkpoint")
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument("--output", required=True, help="Path to output TensorRT engine")
    parser.add_argument("--precision", default="FP16", choices=["FP32", "FP16", "INT8"], 
                       help="TensorRT precision (default: FP16)")
    
    args = parser.parse_args()
    
    success = convert_to_tensorrt(
        args.checkpoint,
        args.config,
        args.output,
        args.precision
    )
    
    sys.exit(0 if success else 1)
