import os
import sys
import shutil
import numpy as np
import librosa
import soundfile as sf
import torch

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from hf_utils import load_custom_model_from_hf

def slice_audio(file_path, output_dir, chunk_len=5, overlap=1):
    y, sr = librosa.load(file_path, sr=44100, mono=True)
    chunk_samples = int(chunk_len * sr)
    hop_samples = int((chunk_len - overlap) * sr)
    
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    count = 0
    
    for start in range(0, len(y) - chunk_samples + 1, hop_samples):
        chunk = y[start : start + chunk_samples]
        if np.abs(chunk).mean() > 0.01: # Skip silence
            out_path = os.path.join(output_dir, f"{base_name}_{count:03d}.wav")
            sf.write(out_path, chunk, sr)
            count += 1
    return count

def prepare_dataset():
    print("--- Preparing Rapper Dataset ---")
    
    source_dir = "datasets/rapper"
    target_dir = "datasets/rapper_finetune"
    
    if not os.path.exists(source_dir):
        print(f"Creating placeholder directory: {source_dir}")
        os.makedirs(source_dir)
        print("Please place rapper audio files in this folder and run again.")
        return

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    
    # 1. Process Files
    print(f"Scanning {source_dir}...")
    files = [f for f in os.listdir(source_dir) if f.endswith('.wav') or f.endswith('.mp3')]
    
    if not files:
        print("No audio files found. Please add .wav or .mp3 files.")
        return
        
    total_chunks = 0
    for f in files:
        src = os.path.join(source_dir, f)
        print(f"Processing {f}...")
        total_chunks += slice_audio(src, target_dir)
        
    print(f"Generated {total_chunks} clips in {target_dir}")
    
    # 2. Generate Training Script
    # Determine which base model to use.
    # If Russian fine-tune exists, we might want to start from THERE?
    # The user asked: "fine tune the model that will be created with for example a rapper voice"
    # This implies 2-stage: Original -> Russian -> Rapper? OR Original -> Rapper?
    # User said: "metrics for original model, russian fine-tuning model and rapper russian fine-tuning model"
    # This implies "rapper russian" is trained ON TOP OF "russian".
    
    # We'll default to the Russian checkpoint if it exists, otherwise Base.
    
    russian_ckpt = "runs/russian_finetune_small/ft_model.pth"
    use_russian_base = os.path.exists(russian_ckpt)
    
    print("\nLocating Base Model...")
    try:
        base_model, config_path = load_custom_model_from_hf(
            "Plachta/Seed-VC", 
            "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth", 
            "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
        )
        
        pretrained_path = russian_ckpt if use_russian_base else base_model
        
        batch_content = f"""@echo off
echo ==================================================
echo      Seed-VC: Rapper Voice Fine-tuning
echo ==================================================
echo.
echo Base Model: {pretrained_path}
echo Dataset: {target_dir}
echo.

REM Run from project root
python train.py ^
    --config "{config_path}" ^
    --pretrained-ckpt "{pretrained_path}" ^
    --dataset-dir "{target_dir}" ^
    --run-name "rapper_russian_finetune" ^
    --batch-size 4 ^
    --max-epochs 50 ^
    --save-every 100 ^
    --num-workers 0

echo.
echo Training Finished!
echo Checkpoint: runs/rapper_russian_finetune/ft_model.pth
pause
"""
        batch_file = os.path.join(os.path.dirname(__file__), "train_rapper.bat")
        with open(batch_file, "w") as f:
            f.write(batch_content)
        print(f"Created training script: {batch_file}")
        if use_russian_base:
            print("(Configured to continue from Russian Fine-tuned model)")
        else:
            print("(Configured to use original Pre-trained model - Russian model not found)")

    except Exception as e:
        print(f"Error locating model: {e}")

if __name__ == "__main__":
    prepare_dataset()
