# Jetson Nano Edge Optimization for Seed-VC (Russian FT Model)

This folder contains tools and scripts to optimize the Seed-VC model (specifically the Russian fine-tuned version) for deployment on NVIDIA Jetson Nano.

## Contents

- `models.py`: Core optimization utilities (Pruning, Quantization, Mixed Precision).
- `benchmark.py`: Micro-benchmarking script for measuring latency and model size of the DiT component.
- `find_optimal_config.py`: Systematic search for the best combination of pruning ratio and quantization strategy.
- `mixed_precision_quant.py`: logic for applying different precisions to Attention vs Feed-Forward layers.

## Optimization Strategies

### 1. Structured Pruning
We use `torch.nn.utils.prune.ln_structured` to remove entire channels/filters. This is more GPU-friendly than unstructured pruning and leads to actual inference speedups on Jetson Nano.
- Tested ratios: 20%, 30%, 40%, 50%.

### 2. Quantization (FP16, INT8, INT4)
- **FP16**: Half precision is the baseline for Jetson Nano (Maxwell/Pascal GPUs).
- **INT8 PTQ**: Post-Training Static Quantization. Best for CPU-offloaded tasks or when converted to TensorRT.
- **QAT (Quantization-Aware Training)**: Scripts provided to prepare the model for QAT if further fine-tuning is desired.
- **INT4**: Conceptual implementation provided (requires specialized kernels for hardware speedup).

### 3. Mixed Precision
- **Attention Layers**: FP16 (to preserve quality in key/query/value mappings).
- **Feed-Forward Layers**: INT8 (where weight distribution is often more uniform).

## How to Run Benchmarks on Jetson Nano

1. **Setup Environment**: Ensure you are in the Jetson-compatible virtual environment (see `JETSON_NANO_SETUP.md`).
2. **Run Benchmarker**:
   ```bash
   python project/edge_optimization/benchmark.py
   ```
3. **Run Optimal Config Search**:
   ```bash
   python project/edge_optimization/find_optimal_config.py
   ```

## Expected Outcome

The goal is to achieve **<200ms latency** on the Jetson Nano with a model size **<500MB** while maintaining high speaker similarity and speech intelligibility (STOI).

Based on research, the **30% Pruned + FP16** configuration is usually the optimal balance for this architecture.
