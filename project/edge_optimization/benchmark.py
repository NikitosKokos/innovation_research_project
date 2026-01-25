import os
import sys
import torch
import time
import numpy as np
import librosa
import shutil
from tabulate import tabulate

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from project.edge_optimization.models import OptimizationUtils, get_model_size
from seed_vc_wrapper import SeedVCWrapper
from project.preprocessing.evaluation import compute_pesq, compute_stoi
from modules.commons import build_model, load_checkpoint, recursive_munch
import yaml

class JetsonBenchmarker:
    def __init__(self, checkpoint_path, config_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        
        print(f"Loading model for benchmarking: {os.path.basename(checkpoint_path)}")
        self.config = yaml.safe_load(open(config_path, 'r'))
        self.model_params = recursive_munch(self.config['model_params'])
        
        # We only benchmark the DiT component as it's the main bottleneck
        self.base_model = build_model(self.model_params, stage='DiT')
        self.base_model, _, _, _ = load_checkpoint(
            self.base_model, None, checkpoint_path,
            load_only_params=True, ignore_modules=[], is_distributed=False
        )
        # DiT model is usually in 'cfm' or 'estimator'
        self.dit_model = self.base_model.cfm.estimator
        
        # Remove weight norm before benchmarking to allow deepcopy and avoid overhead
        OptimizationUtils.remove_weight_norm(self.dit_model)
        
        self.dit_model.eval().to(self.device)
        
    def benchmark_latency(self, model, num_runs=50, input_shape=None):
        """Measures average latency in ms using high-precision timing"""
        # Get dimensions from config
        in_channels = self.config['model_params']['DiT'].get('in_channels', 80)
        content_dim = self.config['model_params']['DiT'].get('content_dim', 512)
        style_dim = self.config['model_params']['style_encoder'].get('dim', 192)
        T = 192 # default sequence length
        
        if input_shape is None:
            input_shape = (1, in_channels, T)
        
        # Detect model device safely
        model_device = torch.device('cpu')
        try:
            model_device = next(model.parameters()).device
        except (StopIteration, Exception):
            # Try buffers if no parameters (common in quantized models)
            try:
                model_device = next(model.buffers()).device
            except:
                pass

        # Create dummy inputs on the same device as the model
        x = torch.randn(input_shape).to(model_device).contiguous()
        prompt_x = torch.randn(input_shape).to(model_device).contiguous()
        x_lens = torch.tensor([T]).to(model_device)
        t = torch.tensor([0.5]).to(model_device)
        style = torch.randn(1, style_dim).to(model_device).contiguous()
        cond = torch.randn(1, T, content_dim).to(model_device).contiguous()
        
        # Handle precision
        try:
            param = next(model.parameters())
            if param.dtype in [torch.float16, torch.half]:
                x, prompt_x, style, cond, t = x.half(), prompt_x.half(), style.half(), cond.half(), t.half()
        except (StopIteration, Exception):
            pass
        
        # Initialize caches for DiT if required
        if hasattr(model, 'setup_caches'):
            model.setup_caches(max_batch_size=1, max_seq_length=T)

        # Warmup
        model.eval()
        for _ in range(10):
            with torch.no_grad():
                _ = model(x, prompt_x, x_lens, t, style, cond)
        
        if model_device.type == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(num_runs):
                with torch.no_grad():
                    _ = model(x, prompt_x, x_lens, t, style, cond)
            end_event.record()
            torch.cuda.synchronize()
            return start_event.elapsed_time(end_event) / num_runs
        else:
            # CPU timing
            start_time = time.time()
            for _ in range(num_runs):
                with torch.no_grad():
                    _ = model(x, prompt_x, x_lens, t, style, cond)
            return ((time.time() - start_time) / num_runs) * 1000

    def run_all_tests(self):
        results = []
        
        # Test Cases
        test_cases = [
            ("FP32 (Baseline)", lambda m: m, 0.0, False),
            ("FP16 (Half)", lambda m: m.half(), 0.0, True),
            ("Pruned 20%", lambda m: OptimizationUtils.apply_structured_pruning(copy_model(m), 0.2), 0.2, False),
            ("Pruned 30%", lambda m: OptimizationUtils.apply_structured_pruning(copy_model(m), 0.3), 0.3, False),
            ("Pruned 50%", lambda m: OptimizationUtils.apply_structured_pruning(copy_model(m), 0.5), 0.5, False),
            ("INT8 PTQ (CPU)", lambda m: self.get_ptq_model(m), 0.0, False), # PTQ usually on CPU for benchmarking if no TensorRT
        ]

        def copy_model(m):
            import copy
            return copy.deepcopy(m)

        for name, transform, p_ratio, is_half in test_cases:
            print(f"\nRunning test: {name}...")
            try:
                test_model = transform(self.dit_model)
                if is_half:
                    test_model = test_model.half()
                
                size = get_model_size(test_model)
                
                # Use CPU for INT8 PTQ as PyTorch CUDA quantization is limited
                dev = 'cpu' if "INT8" in name else 'cuda'
                test_model.to(dev)
                
                # Mock input shape for DiT
                # DiT small: hidden_dim=512
                latency = self.benchmark_latency(test_model, num_runs=20)
                
                results.append([name, f"{latency:.2f}ms", f"{size:.2f}MB", "Pending", "OK"])
            except Exception as e:
                print(f"Error in {name}: {e}")
                results.append([name, "N/A", "N/A", "N/A", f"Error: {str(e)[:30]}"])

        return results

    def get_ptq_model(self, model):
        """
        Convert model to INT8 using Dynamic Quantization.
        """
        try:
            print("Attempting Dynamic Quantization (INT8)...")
            # Move to CPU explicitly
            model_cpu = copy.deepcopy(model).to('cpu')
            model_cpu.eval()
            
            # Dynamic quantization is more robust for Transformers
            quantized = torch.quantization.quantize_dynamic(
                model_cpu, 
                {torch.nn.Linear}, 
                dtype=torch.qint8
            )
            return quantized
        except Exception as e:
            print(f"Quantization failed: {e}")
            import traceback
            traceback.print_exc()
            raise e

def main():
    print("=== Jetson Nano Edge Optimization Benchmarking ===")
    
    ckpt = "runs/russian_finetune_small_v3/ft_model.pth"
    config = "configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
    
    if not os.path.exists(ckpt):
        print(f"Error: Checkpoint not found at {ckpt}")
        return

    benchmarker = JetsonBenchmarker(ckpt, config)
    results = benchmarker.run_all_tests()
    
    headers = ["Strategy", "Latency", "Size", "Quality (STOI)", "Status"]
    print("\n" + tabulate(results, headers=headers, tablefmt="grid"))
    
    print("\nNotes:")
    print("1. Latency measured on DiT component (main bottleneck).")
    print("2. FP16 is highly recommended for Jetson Nano (Maxwell/Pascal GPUs).")
    print("3. INT8 PTQ speedup is most visible on CPU; for GPU use TensorRT conversion.")
    print("4. Quality metrics (STOI) require full inference loop (not included in this micro-benchmark).")

import copy
if __name__ == "__main__":
    main()
