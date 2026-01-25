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
from project.preprocessing.denoising import WienerDenoiser
from project.preprocessing.normalization import LUFSNormalizer

def process_and_slice_audio(file_path, output_dir, target_sr=44100, min_len=3, max_len=15):
    print(f"Loading {os.path.basename(file_path)}...")
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    
    # 1. Denoise
    print("Applying denoising (Wiener filter)...")
    denoiser = WienerDenoiser(sr=target_sr)
    y = denoiser.denoise(y)
    
    # 2. Normalize (LUFS)
    print("Applying LUFS normalization...")
    normalizer = LUFSNormalizer(target_lufs=-23.0, sr=target_sr)
    y = normalizer.normalize(y)
    
    # 3. Slice based on energy (Simple VAD)
    print(f"Slicing into {min_len}-{max_len}s chunks...")
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # top_db=45 is better for rap which has a high dynamic range
    intervals = librosa.effects.split(y, top_db=45)
    
    chunk_count = 0
    for start_idx, end_idx in intervals:
        segment = y[start_idx:end_idx]
        duration = len(segment) / target_sr
        
        if duration < min_len:
            continue
            
        # If segment is too long, split it further
        if duration > max_len:
            num_sub_chunks = int(np.ceil(duration / 10.0)) # target ~10s sub-chunks
            sub_chunk_samples = len(segment) // num_sub_chunks
            for i in range(num_sub_chunks):
                sub_segment = segment[i*sub_chunk_samples : (i+1)*sub_chunk_samples]
                if len(sub_segment) / target_sr >= min_len:
                    out_path = os.path.join(output_dir, f"{base_name}_{chunk_count:04d}.wav")
                    sf.write(out_path, sub_segment, target_sr)
                    chunk_count += 1
        else:
            out_path = os.path.join(output_dir, f"{base_name}_{chunk_count:04d}.wav")
            sf.write(out_path, segment, target_sr)
            chunk_count += 1
            
    return chunk_count

def prepare_dataset():
    print("--- Preparing Oxxxymiron Rapper Dataset ---")
    
    source_file = "datasets/oxxxymiron_dataset_raw.wav"
    target_dir = "datasets/rapper_finetune"
    
    if not os.path.exists(source_file):
        # Check if it's in the root
        if os.path.exists("oxxxymiron_dataset_raw.wav"):
            source_file = "oxxxymiron_dataset_raw.wav"
        else:
            print(f"Error: {source_file} not found. Please ensure the raw audio is present.")
            return

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    
    total_chunks = process_and_slice_audio(source_file, target_dir)
    print(f"\nSuccess! Generated {total_chunks} high-quality clips in {target_dir}")
    
    # 2. Generate Training Script
    russian_ckpt = "runs/russian_finetune_small_v3/ft_model.pth"
    if not os.path.exists(russian_ckpt):
        # Fallback to older version if v3 isn't there
        russian_ckpt = "runs/russian_finetune_small/ft_model.pth"
        
    use_russian_base = os.path.exists(russian_ckpt)
    
    try:
        # Get original config for reference
        _, config_path = load_custom_model_from_hf(
            "Plachta/Seed-VC", 
            "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth", 
            "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
        )
        
        pretrained_path = russian_ckpt if use_russian_base else "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"
        
        batch_content = f"""@echo off
echo ==================================================
echo      Seed-VC: Rapper (Oxxxymiron) Fine-tuning
echo ==================================================
echo.
echo Base Model: {pretrained_path}
echo Dataset: {target_dir}
echo GPU: RTX 4060 (Optimized)
echo.

REM Run from project root
python train.py ^
    --config "{config_path}" ^
    --pretrained-ckpt "{pretrained_path}" ^
    --dataset-dir "{target_dir}" ^
    --run-name "rapper_oxxxy_finetune" ^
    --batch-size 4 ^
    --max-epochs 100 ^
    --save-every 200 ^
    --num-workers 2

echo.
echo Training Finished!
echo Checkpoint: runs/rapper_oxxxy_finetune/ft_model.pth
pause
"""
        batch_file = os.path.join(os.path.dirname(__file__), "train_rapper.bat")
        with open(batch_file, "w") as f:
            f.write(batch_content)
        print(f"Created training script: {batch_file}")
        
    except Exception as e:
        print(f"Error generating script: {e}")

if __name__ == "__main__":
    prepare_dataset()
