import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.quantization as quantization
import os
import time
import copy
from typing import Dict, Any, List

class OptimizationUtils:
    @staticmethod
    def remove_weight_norm(model: nn.Module):
        """
        Recursively remove weight normalization from all modules.
        This is necessary before deepcopying modules that use weight_norm.
        """
        from torch.nn.utils import remove_weight_norm
        for name, module in model.named_modules():
            try:
                remove_weight_norm(module)
            except (ValueError, AttributeError):
                # Module doesn't have weight_norm or it's already removed
                pass
        return model

    @staticmethod
    def apply_structured_pruning(model: nn.Module, amount: float = 0.3):
        """
        Apply structured pruning to Conv2d and Linear layers.
        Removes entire channels/filters.
        """
        print(f"Applying structured pruning (amount={amount})...")
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
                prune.remove(module, 'weight')
            elif isinstance(module, nn.Linear):
                # For Linear layers, we prune based on rows
                prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
                prune.remove(module, 'weight')
        return model

    @staticmethod
    def prepare_ptq(model: nn.Module, backend='qnnpack'):
        """
        Prepare model for Post-Training Static Quantization.
        Note: Jetson Nano uses qnnpack for ARM.
        """
        model.eval()
        model.qconfig = quantization.get_default_qconfig(backend)
        model_prepared = quantization.prepare(model)
        return model_prepared

    @staticmethod
    def convert_ptq(model_prepared: nn.Module):
        """
        Convert prepared model to quantized version.
        """
        return quantization.convert(model_prepared)

    @staticmethod
    def prepare_qat(model: nn.Module, backend='qnnpack'):
        """
        Prepare model for Quantization-Aware Training.
        """
        model.train()
        model.qconfig = quantization.get_default_qat_qconfig(backend)
        model_prepared = quantization.prepare_qat(model)
        return model_prepared

class PrunedModelWrapper(nn.Module):
    def __init__(self, model, pruning_ratio=0.3):
        super().__init__()
        # Ensure weight norm is removed before deepcopy
        OptimizationUtils.remove_weight_norm(model)
        self.model = copy.deepcopy(model)
        OptimizationUtils.apply_structured_pruning(self.model, pruning_ratio)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def setup_caches(self, *args, **kwargs):
        if hasattr(self.model, 'setup_caches'):
            return self.model.setup_caches(*args, **kwargs)

class MixedPrecisionModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, *args, **kwargs):
        with torch.cuda.amp.autocast():
            return self.model(*args, **kwargs)

    def setup_caches(self, *args, **kwargs):
        if hasattr(self.model, 'setup_caches'):
            return self.model.setup_caches(*args, **kwargs)

def get_model_size(model, is_pruned=False):
    """Returns model size in MB. If is_pruned, calculates size based on non-zero parameters."""
    if not is_pruned:
        # For quantized models, standard state_dict saving is more reliable
        try:
            torch.save(model.state_dict(), "temp.p")
            size = os.path.getsize("temp.p") / (1024 * 1024)
            os.remove("temp.p")
            return size
        except Exception:
            # Fallback for models that can't be easily saved
            total_params = sum(p.numel() for p in model.parameters())
            return (total_params * 4) / (1024 * 1024)
    else:
        # Calculate size based on non-zero elements and their actual dtype
        total_nonzero = 0
        bytes_per_param = 4 # Default to FP32
        
        params = list(model.parameters())
        if not params:
            # Handle quantized models with no parameters
            torch.save(model.state_dict(), "temp.p")
            size = os.path.getsize("temp.p") / (1024 * 1024)
            os.remove("temp.p")
            return size

        for param in params:
            total_nonzero += torch.count_nonzero(param).item()
            # Use the dtype of the first parameter we find
            if bytes_per_param == 4:
                if param.dtype == torch.float16 or param.dtype == torch.half:
                    bytes_per_param = 2
                elif param.dtype == torch.int8:
                    bytes_per_param = 1
        
        return (total_nonzero * bytes_per_param) / (1024 * 1024)
