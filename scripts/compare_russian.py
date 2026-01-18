import os
import torch
import torchaudio
import librosa
import argparse
from inference import voice_conversion
from modules.commons import build_model, recursive_munch, load_checkpoint
import yaml

def compare_models(source, target, original_ckpt, finetuned_ckpt, config_path, output_dir="audio_outputs/comparison"):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Comparing models on source: {source}")
    print(f"Reference: {target}")
    
    # Common settings
    diffusion_steps = 10
    length_adjust = 1.0
    inference_cfg_rate = 0.7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Run with Original Model
    print("\n--- Running Original Model ---")
    os.system(f"python inference.py --source \"{source}\" --target \"{target}\" --output \"{output_dir}\" --checkpoint \"{original_ckpt}\" --config \"{config_path}\" --diffusion-steps {diffusion_steps} --inference-cfg-rate {inference_cfg_rate} --fp16 True")
    
    # Rename output
    original_out = os.path.join(output_dir, os.path.basename(source).replace(".wav", f"_vc_{os.path.basename(target).replace('.wav', '')}_{length_adjust}_{diffusion_steps}_{inference_cfg_rate}.wav"))
    if os.path.exists(original_out):
        new_name = os.path.join(output_dir, "result_original.wav")
        if os.path.exists(new_name): os.remove(new_name)
        os.rename(original_out, new_name)
    
    # 2. Run with Fine-tuned Model
    print("\n--- Running Fine-tuned Model ---")
    # Check if fine-tuned checkpoint exists
    if not os.path.exists(finetuned_ckpt):
        print(f"Error: Fine-tuned checkpoint not found at {finetuned_ckpt}")
        print("Did you finish training?")
        return

    os.system(f"python inference.py --source \"{source}\" --target \"{target}\" --output \"{output_dir}\" --checkpoint \"{finetuned_ckpt}\" --config \"{config_path}\" --diffusion-steps {diffusion_steps} --inference-cfg-rate {inference_cfg_rate} --fp16 True")
    
    # Rename output
    if os.path.exists(original_out): # Inference.py generates same name
        new_name = os.path.join(output_dir, "result_finetuned.wav")
        if os.path.exists(new_name): os.remove(new_name)
        os.rename(original_out, new_name)

    print(f"\nComparison generation done.")
    print(f"Original: {os.path.join(output_dir, 'result_original.wav')}")
    print(f"Fine-tuned: {os.path.join(output_dir, 'result_finetuned.wav')}")

if __name__ == "__main__":
    # Example usage
    # Adjust paths as needed
    SOURCE = "audio_inputs/user/input_yo.wav" # Replace with your rapper recording
    TARGET = "audio_inputs/reference/ref.wav" # Replace with rapper reference
    
    # Original checkpoint (auto-downloaded usually, but specify path if known)
    # If using auto-download, inference.py handles it if arg is empty, but we need to control it here.
    # Assuming standard path:
    ORIGINAL = "DiT_uvit_tat_xlsr_ema.pth" 
    
    # Your new checkpoint
    # After running run_manual_finetune.bat, it will be here
    FINETUNED = "runs/russian_finetune_manual/ft_model.pth" 
    
    CONFIG = "configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml"
    
    compare_models(SOURCE, TARGET, ORIGINAL, FINETUNED, CONFIG)
