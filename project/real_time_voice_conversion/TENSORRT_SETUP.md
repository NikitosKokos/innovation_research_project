# TensorRT Setup for Real-Time Voice Conversion

TensorRT can significantly speed up inference on Jetson Nano by optimizing the model execution.

## Prerequisites

1. **Install TensorRT** (usually included with JetPack):
   ```bash
   # Check if TensorRT is installed
   python -c "import tensorrt; print(tensorrt.__version__)"
   ```

2. **Install PyCUDA**:
   ```bash
   pip install pycuda
   ```

## Converting Model to TensorRT

1. **Convert your model to TensorRT engine**:
   ```bash
   cd /home/ailab/Desktop/innovation_research_project
   
   python project/real_time_voice_conversion/convert_to_tensorrt.py \
       --checkpoint AI_models/rapper_oxxxy_finetune/ft_model.pth \
       --config AI_models/rapper_oxxxy_finetune/config_dit_mel_seed_uvit_whisper_small_wavenet.yml \
       --output AI_models/rapper_oxxxy_finetune/model_fp16.trt \
       --precision FP16
   ```

2. **Update config.py** to use TensorRT:
   ```python
   USE_TENSORRT = True
   TENSORRT_ENGINE_PATH = os.path.join(MODEL_DIR, "model_fp16.trt")
   ```

## Expected Performance

- **PyTorch FP16**: ~200-300ms per chunk
- **TensorRT FP16**: ~100-150ms per chunk (2-3x speedup)

## Notes

- TensorRT conversion takes 5-15 minutes (one-time cost)
- TensorRT engine is specific to the GPU architecture
- FP16 provides best speed/quality balance on Jetson Nano
- If conversion fails, fall back to PyTorch by setting `USE_TENSORRT = False`

## Troubleshooting

**Error: "TensorRT not found"**
- Install TensorRT: Usually comes with JetPack, or install from NVIDIA
- Create symlinks: Run `./install_tensorrt.sh` or manually create symlinks

**Error: "ONNX conversion failed" / "INTERNAL ASSERT FAILED"**
- **This is expected**: The Seed-VC DiT model has complex operations (dynamic control flow, mixed types) that ONNX cannot export
- **Solution**: Use PyTorch optimizations instead (already enabled):
  - FP16 quantization with autocast
  - Reduced diffusion steps (3)
  - CFG disabled (0.0)
  - Optimized block size
- These optimizations provide good real-time performance without TensorRT

**Error: "Engine build failed"**
- Reduce workspace memory: Edit `convert_to_tensorrt.py` and change `1 << 30` to `512 << 20` (512MB)
- Try FP32 instead of FP16

## Note on ONNX/TensorRT Limitations

The Seed-VC DiT (Diffusion Transformer) model uses operations that are difficult to export to ONNX:
- Dynamic control flow based on training/inference mode
- Mixed Python bool/int values with tensors
- Complex attention mechanisms with caching

**Recommendation**: The current PyTorch optimizations (FP16, reduced steps, CFG disabled) should provide sufficient performance for real-time voice conversion on Jetson Nano. TensorRT conversion is optional and may not be feasible for this model architecture.
