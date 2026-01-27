import torch
import torch.onnx
import os
import sys
import yaml
import argparse

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from modules.commons import build_model, load_checkpoint, recursive_munch

def convert_to_onnx(checkpoint_path: str, config_path: str, output_onnx_path: str):
    print(f"Loading model from {checkpoint_path}")
    
    # Load model config
    config = yaml.safe_load(open(config_path, 'r'))
    model_params = recursive_munch(config['model_params'])
    
    # Build full model
    model = build_model(model_params, stage='DiT')
    
    # Load checkpoint - should be in format {'net': {'cfm': state_dict}}
    model, _, _, _ = load_checkpoint(
        model, None, checkpoint_path,
        load_only_params=True, ignore_modules=[], is_distributed=False
    )
    
    # Extract DiT component
    dit_model = model.cfm.estimator
    dit_model.eval()
    
    # Set to inference mode (disables dropout, batch norm updates, etc.)
    torch.set_grad_enabled(False)
    
    # Initialize caches (required for DiT model)
    if hasattr(dit_model, 'setup_caches'):
        dit_model.setup_caches(max_batch_size=1, max_seq_length=8192)

    # Create a wrapper for ONNX export that handles problematic operations
    class DiTWrapper(torch.nn.Module):
        def __init__(self, dit_model):
            super().__init__()
            self.dit_model = dit_model
            # Ensure model is in eval mode
            self.dit_model.eval()
            for param in self.dit_model.parameters():
                param.requires_grad = False
        
        def forward(self, x, prompt_x, x_lens, t, style, cond):
            # Always set mask_content=False for ONNX export (inference mode)
            # Ensure model is in eval mode during forward
            self.dit_model.eval()
            with torch.no_grad():
                return self.dit_model(x, prompt_x, x_lens, t, style, cond, mask_content=False)
    
    wrapped_model = DiTWrapper(dit_model)
    wrapped_model.eval()
    
    # Disable all gradient computation
    for param in wrapped_model.parameters():
        param.requires_grad = False

    # Dummy inputs for ONNX export
    in_channels = config['model_params']['DiT'].get('in_channels', 80)
    content_dim = config['model_params']['DiT'].get('content_dim', 512)
    style_dim = config['model_params']['style_encoder'].get('dim', 192)
    T = 192 # Arbitrary sequence length for export

    dummy_x = torch.randn(1, in_channels, T)
    dummy_prompt_x = torch.randn(1, in_channels, T)
    dummy_x_lens = torch.tensor([T], dtype=torch.long)
    dummy_t = torch.tensor([0.5], dtype=torch.float32)
    dummy_style = torch.randn(1, style_dim)
    dummy_cond = torch.randn(1, T, content_dim)

    print("Exporting model to ONNX (this may take a few minutes)...")
    print("[Info] Note: ONNX export may fail due to model complexity. TensorRT conversion is optional.")
    
    # Export the model with additional options to handle edge cases
    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            (dummy_x, dummy_prompt_x, dummy_x_lens, dummy_t, dummy_style, dummy_cond),
            output_onnx_path,
            input_names=['x', 'prompt_x', 'x_lens', 't', 'style', 'cond'],
            output_names=['output'],
            opset_version=17, # Increased to support complex operations (RoPE)
            do_constant_folding=True,
            export_params=True,
            verbose=False,
            dynamic_axes={
                'x': {2: 'T'},
                'prompt_x': {2: 'T'},
                'x_lens': {0: 'batch_size'},
                'cond': {1: 'T'}
            },
            # Additional options to handle problematic operations
            training=torch.onnx.TrainingMode.EVAL,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        )
    print(f"[Success] Model successfully converted to ONNX: {output_onnx_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PyTorch Seed-VC model to ONNX format.')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the PyTorch model checkpoint (.pth).')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the model configuration YAML file.')
    parser.add_argument('--output', type=str, default='seed_vc.onnx',
                        help='Output path for the ONNX model.')
    
    args = parser.parse_args()

    convert_to_onnx(args.checkpoint, args.config, args.output)
