# Edge Optimization for Seed-VC on NVIDIA Jetson Nano

This directory contains comprehensive optimization tools for deploying Seed-VC on edge devices like NVIDIA Jetson Nano.

## Overview

The optimization system answers two key research questions:

1. **How do different quantization methods affect performance?**
   - FP16 quantization
   - INT8 Post-Training Quantization (PTQ)
   - INT8 Quantization-Aware Training (QAT)
   - INT4 QAT (simulated)
   - Mixed-precision (Attention FP16 + Feed-forward INT8)

2. **What is the optimal pruning ratio with mixed-precision quantization?**
   - Tests pruning ratios: 20%, 30%, 40%, 50%
   - Combined with various quantization strategies

## Metrics Evaluated

### Performance Metrics
- **Inference Latency**: Measured in milliseconds (ms) for DiT component
- **Model Size**: Size in MB and compression ratio vs baseline
- **Size Reduction**: Percentage reduction from baseline FP32 model

### Quality Metrics
- **Speaker Similarity (SECS)**: Speaker Embedding Cosine Similarity (0-1, higher is better)
- **Pitch Preservation**:
  - F0 Correlation: Correlation between source and converted F0 (-1 to 1, higher is better)
  - F0 RMSE: Root Mean Square Error of F0 difference (lower is better, in Hz)
- **STOI Score**: Short-Time Objective Intelligibility (0-1, higher is better)

## Files

- `find_optimal_config.py`: Main comprehensive optimization script
- `benchmark.py`: Latency benchmarking utilities
- `models.py`: Model optimization utilities (pruning, quantization)
- `mixed_precision_quant.py`: Mixed-precision quantization helpers
- `quality_metrics.py`: Quality evaluation metrics (SECS, pitch, STOI)

## Usage

### Basic Usage

```bash
cd project/edge_optimization
python find_optimal_config.py
```

### With Custom Paths

```bash
python find_optimal_config.py \
    --checkpoint AI_models/rapper_oxxxy_finetune/ft_model.pth \
    --config AI_models/rapper_oxxxy_finetune/config_dit_mel_seed_uvit_whisper_small_wavenet.yml \
    --test-audio path/to/test.wav \
    --reference-audio audio_inputs/reference/ref01_processed.wav \
    --device cuda \
    --output results.json
```

## Optimization Strategies Tested

### 1. FP16 Quantization
- **Pros**: 2x memory reduction, hardware acceleration on Jetson
- **Cons**: Slight quality degradation possible
- **Best for**: GPU inference on Jetson Nano

### 2. INT8 Post-Training Quantization (PTQ)
- **Pros**: 4x memory reduction, good for CPU inference
- **Cons**: Requires calibration data, may have quality loss
- **Best for**: CPU-only inference or when memory is critical

### 3. INT8 Quantization-Aware Training (QAT)
- **Pros**: Better quality than PTQ, 4x memory reduction
- **Cons**: Requires retraining/fine-tuning
- **Best for**: Production deployments with training budget

### 4. INT4 QAT
- **Pros**: 8x memory reduction (theoretical)
- **Cons**: Significant quality loss, requires custom kernels
- **Best for**: Extreme memory constraints (not recommended for voice conversion)

### 5. Mixed-Precision Quantization
- **Strategy**: Attention layers in FP16, Feed-forward layers in INT8
- **Pros**: Balance between quality and performance
- **Cons**: Requires careful implementation (TensorRT recommended)
- **Best for**: Optimal quality/speed tradeoff

### 6. Pruning
- **Ratios Tested**: 20%, 30%, 40%, 50%
- **Method**: Structured pruning (removes entire channels/filters)
- **Impact**: Reduces model size and inference time, may affect quality

## Expected Results

### Typical Performance on Jetson Nano

| Configuration | Latency (ms) | Size (MB) | Size Reduction | Quality Impact |
|--------------|--------------|-----------|----------------|----------------|
| Baseline FP32 | ~800-1200 | ~200-300 | 0% | Reference |
| FP16 | ~400-600 | ~100-150 | ~50% | Minimal |
| 30% Pruned + FP16 | ~300-450 | ~70-105 | ~65% | Small |
| INT8 PTQ | ~600-900 | ~50-75 | ~75% | Moderate |
| Mixed-Precision | ~350-500 | ~80-120 | ~60% | Small |

### Quality Metrics Targets

- **SECS**: >0.85 (excellent), >0.80 (good), >0.75 (acceptable)
- **F0 Correlation**: >0.90 (excellent), >0.85 (good), >0.80 (acceptable)
- **F0 RMSE**: <30 Hz (excellent), <50 Hz (good), <80 Hz (acceptable)
- **STOI**: >0.90 (excellent), >0.85 (good), >0.80 (acceptable)

## Output

The script generates:

1. **Console Output**: Formatted table with all results
2. **JSON File**: Detailed results including:
   - All tested configurations
   - Performance and quality metrics
   - Optimal configuration recommendation

### Example Output Structure

```json
{
  "baseline_size_mb": 250.5,
  "results": [
    {
      "config_name": "Prune30%_FP16",
      "quantization_type": "FP16",
      "pruning_ratio": 0.3,
      "latency_ms": 425.3,
      "model_size_mb": 87.5,
      "size_reduction_ratio": 65.1,
      "speaker_similarity": 0.852,
      "f0_correlation": 0.891,
      "f0_rmse": 28.5,
      "stoi_score": 0.876,
      "success": true
    }
  ],
  "optimal_config": { ... }
}
```

## Recommendations for Jetson Nano

Based on typical results:

1. **Best Overall**: 30% Pruning + FP16
   - Good balance of speed, size, and quality
   - Leverages Jetson's FP16 acceleration

2. **Maximum Speed**: 40% Pruning + FP16
   - Fastest inference
   - Acceptable quality for most use cases

3. **Maximum Compression**: INT8 PTQ + 50% Pruning
   - Smallest model size
   - Best for memory-constrained scenarios

4. **Best Quality**: FP16 only (no pruning)
   - Highest quality preservation
   - Still 2x memory reduction

## Limitations

1. **Quality Evaluation**: Full quality metrics require running complete inference pipeline, which is time-consuming. The current implementation uses simplified evaluation.

2. **INT4 Support**: True INT4 quantization requires custom kernels (TensorRT) and is not fully supported in PyTorch.

3. **Mixed-Precision**: True mixed-precision (FP16 attention + INT8 FF) requires TensorRT or custom operators. The current implementation is simplified.

4. **QAT**: Quantization-Aware Training requires retraining. The script prepares models but doesn't actually train them.

## Dependencies

```bash
pip install torch torchvision torchaudio
pip install librosa scipy numpy
pip install resemblyzer  # For speaker similarity
pip install pystoi  # For STOI
pip install pesq  # Optional, for PESQ
pip install tabulate  # For formatted tables
```

## Notes

- Optimization can take 30-60 minutes depending on device
- Results vary based on hardware, temperature, and system load
- For production, consider TensorRT for additional optimizations
- Quality metrics are approximate; full evaluation requires complete inference pipeline

## Future Improvements

- [ ] Full inference pipeline for accurate quality metrics
- [ ] TensorRT integration for true mixed-precision
- [ ] INT4 quantization with custom kernels
- [ ] Automated hyperparameter search
- [ ] Real-time performance profiling
