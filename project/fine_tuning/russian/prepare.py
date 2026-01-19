import os
import sys
import shutil
import random
import librosa
import soundfile as sf
import torch

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from hf_utils import load_custom_model_from_hf

def prepare_dataset():
    print("--- Preparing Russian Dataset (Common Voice) ---")
    
    source_dir = "datasets/clips"
    target_dir = "datasets/russian_finetune"
    max_files = 2000
    
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' not found.")
        return
        
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    
    # 1. Collect Files
    print(f"Scanning {source_dir}...")
    files = [f for f in os.listdir(source_dir) if f.endswith('.mp3') or f.endswith('.wav')]
    print(f"Found {len(files)} files.")
    
    if len(files) > max_files:
        print(f"Randomly selecting {max_files} files...")
        selected = random.sample(files, max_files)
    else:
        selected = files
        
    # 2. Process Files (Copy/Convert)
    print("Processing files...")
    for f in selected:
        src = os.path.join(source_dir, f)
        dst = os.path.join(target_dir, os.path.splitext(f)[0] + ".wav")
        
        # Convert mp3 to wav if needed, or just copy and verify
        try:
            y, sr = librosa.load(src, sr=44100)
            sf.write(dst, y, sr)
        except Exception as e:
            print(f"Failed to process {f}: {e}")
            
    print(f"Dataset ready at: {target_dir}")
    
    # 3. Generate Training Script
    print("\nLocating Base Model...")
    try:
        model_path, config_path = load_custom_model_from_hf(
            "Plachta/Seed-VC", 
            "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth", 
            "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
        )
        
        batch_content = f"""@echo off
echo ==================================================
echo      Seed-VC: Russian Language Fine-tuning
echo ==================================================
echo.
echo Base Model: {model_path}
echo Dataset: {target_dir}
echo.

REM Run from project root
python train.py ^
    --config "{config_path}" ^
    --pretrained-ckpt "{model_path}" ^
    --dataset-dir "{target_dir}" ^
    --run-name "russian_finetune_small_v3" ^
    --batch-size 4 ^
    --max-epochs 20 ^
    --save-every 500 ^
    --num-workers 0

echo.
echo Training Finished!
echo Checkpoint: runs/russian_finetune_small/ft_model.pth
pause
"""
        batch_file = os.path.join(os.path.dirname(__file__), "train_russian.bat")
        with open(batch_file, "w") as f:
            f.write(batch_content)
        print(f"Created training script: {batch_file}")
        
    except Exception as e:
        print(f"Error locating model: {e}")

if __name__ == "__main__":
    prepare_dataset()
