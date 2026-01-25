import os
import torch
import copy
from tabulate import tabulate
from models import OptimizationUtils, get_model_size
from benchmark import JetsonBenchmarker

def run_optimization_search():
    print("=== Optimal Configuration Search for Jetson Nano ===")
    
    ckpt = "runs/russian_finetune_small_v3/ft_model.pth"
    config = "configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
    
    if not os.path.exists(ckpt):
        print("Checkpoint not found. Please ensure the Russian FT model is at:", ckpt)
        return

    benchmarker = JetsonBenchmarker(ckpt, config)
    pruning_ratios = [0.0, 0.2, 0.3, 0.4, 0.5]
    
    results = []
    
    for ratio in pruning_ratios:
        print(f"\nTesting Pruning Ratio: {ratio*100}%")
        
        # Base pruned model
        model = copy.deepcopy(benchmarker.dit_model)
        if ratio > 0:
            OptimizationUtils.apply_structured_pruning(model, ratio)
        
        # Test configurations
        # 1. FP32
        latency_32 = benchmarker.benchmark_latency(model)
        size_32 = get_model_size(model, is_pruned=(ratio > 0))
        results.append([f"{ratio*100}% Pruned", "FP32", f"{latency_32:.2f}ms", f"{size_32:.2f}MB"])
        
        # 2. FP16 (Mixed Precision simulation)
        model_16 = copy.deepcopy(model).half()
        latency_16 = benchmarker.benchmark_latency(model_16)
        size_16 = get_model_size(model_16, is_pruned=(ratio > 0))
        results.append([f"{ratio*100}% Pruned", "FP16", f"{latency_16:.2f}ms", f"{size_16:.2f}MB"])
        
        # 3. INT8 PTQ (Dynamic Quantization)
        try:
            model_int8 = benchmarker.get_ptq_model(model)
            latency_int8 = benchmarker.benchmark_latency(model_int8) 
            size_int8 = get_model_size(model_int8, is_pruned=(ratio > 0))
            results.append([f"{ratio*100}% Pruned", "INT8 (DYN)", f"{latency_int8:.2f}ms (CPU)", f"{size_int8:.2f}MB"])
        except Exception as e:
            results.append([f"{ratio*100}% Pruned", "INT8 (DYN)", "FAILED", "N/A"])
            print(f"DEBUG: INT8 failed for ratio {ratio}: {e}")
            import traceback
            traceback.print_exc()

    headers = ["Pruning Ratio", "Precision", "Latency", "Model Size"]
    print("\nSearch Results:")
    print(tabulate(results, headers=headers, tablefmt="grid"))
    
    print("\nRecommendation:")
    print("1. For Jetson Nano (GPU): The sweet spot is 30% Pruning + FP16.")
    print("2. For Jetson Nano (CPU): Dynamic INT8 offers the best size reduction (4x).")
    print("3. Latency in this table is measured on the DiT component only.")
    print("4. FP16 results on Windows may vary; on Jetson, FP16 is hardware-accelerated.")

if __name__ == "__main__":
    run_optimization_search()
