import torch
import numpy as np
import time

class TensorRTEstimator:
    """
    Wrapper for TensorRT engine to be used as a drop-in replacement for the PyTorch estimator.
    This allows us to reuse the existing CFM inference loop (Euler solver) while accelerating the model forward pass.
    """
    def __init__(self, engine_path):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        print(f"[TensorRT] Loading engine from {engine_path}...")
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # Determine input/output binding indices
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for binding in self.engine:
            idx = self.engine.get_binding_index(binding)
            if self.engine.binding_is_input(binding):
                self.inputs.append({'name': binding, 'index': idx})
            else:
                self.outputs.append({'name': binding, 'index': idx})
                
        # Pre-allocate output buffer (will be resized dynamically if needed, but we start with a guess)
        # Note: In TensorRT with dynamic shapes, output size depends on input size.
        # We will handle allocation in the forward pass.
        self.device_buffers = {}
        
    def __call__(self, x, prompt_x, x_lens, t, style, mu, skip_layers=None):
        """
        Forward pass using TensorRT.
        Matches the signature of the PyTorch DiT model's forward method.
        Note: skip_layers is ignored as TensorRT engines are static graphs.
        """
        import pycuda.driver as cuda
        
        # 1. Prepare Inputs
        # We need to ensure inputs are contiguous and on the host (CPU) or device (GPU) as expected by PyCUDA
        # Since our inputs are already on GPU (from PyTorch), we can use array_interface or data_ptr
        # However, TensorRT python API usually expects GPU pointers.
        
        # Input mapping based on engine definition in convert_to_tensorrt.py:
        # x, prompt_x, x_lens, t, style, cond (mapped from mu)
        
        input_map = {
            'x': x,
            'prompt_x': prompt_x,
            'x_lens': x_lens,
            't': t,
            'style': style,
            'cond': mu
        }
        
        bindings = [None] * self.engine.num_bindings
        
        for item in self.inputs:
            name = item['name']
            tensor = input_map[name]
            
            # Set dynamic shape
            self.context.set_binding_shape(item['index'], tensor.shape)
            
            # Get data pointer (assume tensor is contiguous)
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            
            bindings[item['index']] = tensor.data_ptr()
            
        # 2. Prepare Output
        # We need to calculate output shape based on input shape 'x'
        # Output shape should be same as 'x' (denoised output)
        output_shape = x.shape
        
        # Create output tensor on GPU
        # Check dtype - usually float16 if FP16 mode, but let's check engine or assume same as input
        # For simplicity, we assume output dtype matches input 'x' dtype
        output_tensor = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        
        for item in self.outputs:
            # Assuming single output named 'output'
            bindings[item['index']] = output_tensor.data_ptr()

        # 3. Execute
        # Use execute_async_v2 for dynamic shapes
        self.context.execute_async_v2(bindings=bindings, stream_handle=torch.cuda.current_stream().cuda_stream)
        
        return output_tensor
