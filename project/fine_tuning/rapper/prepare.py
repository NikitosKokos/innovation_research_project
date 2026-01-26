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
# Import advanced preprocessing components
from project.preprocessing.denoising import get_denoiser
from project.preprocessing.normalization import LUFSNormalizer
from project.preprocessing.compression import DynamicRangeCompressor

def process_and_slice_audio(file_path, output_dir, target_sr=44100, min_len=10, max_len=20):
    print(f"Loading {os.path.basename(file_path)}...")
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    
    # --- Advanced Preprocessing Pipeline ---
    
    # 1. Denoise (using noisereduce if available, else Wiener)
    # Note: Noisereduce is better but might be slow on huge files. 
    # Since we are offline preparing, quality > speed.
    print("1. Applying Denoising (noisereduce - gentle mode)...")
    # Reduced aggressiveness handled in get_denoiser now (prop_decrease=0.7)
    denoiser = get_denoiser("noisereduce", sr=target_sr)
    y = denoiser.denoise(y)
    
    # 2. Normalize (LUFS) - Boosted Target
    # -23 LUFS is broadcast standard but often too quiet for training if source was mastered loud
    # -14 LUFS is typical streaming standard (louder, clearer)
    print("2. Applying Normalization (-14 LUFS)...")
    normalizer = LUFSNormalizer(target_lufs=-14.0, sr=target_sr)
    y = normalizer.normalize(y)
    
    # 3. Dynamic Range Compression
    # Finding the "sweet spot" to tame screams without squashing everything
    print("3. Applying Dynamic Range Compression (Ratio 15.0, -10dB Threshold)...")
    compressor = DynamicRangeCompressor(
        threshold_db=-15.0,  # Middle ground between safety and aggressive
        ratio=15.0,          # Firm but not total "brick wall"
        attack_ms=3.0,       # Fast enough to catch transients
        release_ms=50.0,     # Standard release for natural recovery
        sr=target_sr
    )
    y = compressor.compress(y)
    
    # 4. Slice based on energy (Simple VAD)
    print(f"4. Slicing into {min_len}-{max_len}s chunks...")
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # top_db=40 is more relaxed. Keeps almost all natural breaths and tails.
    intervals = librosa.effects.split(y, top_db=40)
    
    chunk_count = 0
    
    # --- FALLBACK LOGIC ---
    total_duration = len(y) / target_sr
    vad_duration = sum([(end - start) for start, end in intervals]) / target_sr
    
    print(f"  - Total Duration: {total_duration:.2f}s")
    print(f"  - VAD Kept Duration: {vad_duration:.2f}s")
    print(f"  - Number of Intervals: {len(intervals)}")
    
    use_fixed_slicing = False
    
    if vad_duration < (total_duration * 0.4) or len(intervals) == 0:
        print(f"Warning: VAD only kept {vad_duration:.1f}s. Switching to fixed slicing.")
        use_fixed_slicing = True
    elif total_duration > 60 and len(intervals) < 3:
         print(f"Warning: Very few segments found. Switching to fixed slicing.")
         use_fixed_slicing = True
        
    if not use_fixed_slicing:
        # Try Standard VAD Slicing
        print("  - Processing VAD intervals with stitching...")
        
        current_segment = []
        current_duration = 0
        
        for i, (start_idx, end_idx) in enumerate(intervals):
            segment = y[start_idx:end_idx]
            duration = len(segment) / target_sr
            
            # Add to current buffer
            current_segment.append(segment)
            current_duration += duration
            
            # If we reached min_len
            if current_duration >= min_len:
                full_segment = np.concatenate(current_segment)
                
                if current_duration > max_len:
                    # Split huge segments
                    num_sub_chunks = int(np.ceil(current_duration / 10.0))
                    sub_chunk_samples = len(full_segment) // num_sub_chunks
                    for k in range(num_sub_chunks):
                        sub_segment = full_segment[k*sub_chunk_samples : (k+1)*sub_chunk_samples]
                        if len(sub_segment) / target_sr >= 2.0:
                            out_path = os.path.join(output_dir, f"{base_name}_{chunk_count:04d}.wav")
                            sf.write(out_path, sub_segment, target_sr)
                            chunk_count += 1
                else:
                    out_path = os.path.join(output_dir, f"{base_name}_{chunk_count:04d}.wav")
                    sf.write(out_path, full_segment, target_sr)
                    chunk_count += 1
                
                # Reset buffer
                current_segment = []
                current_duration = 0
        
        # Handle last remaining buffer if it's long enough
        if current_duration >= min_len:
            full_segment = np.concatenate(current_segment)
            out_path = os.path.join(output_dir, f"{base_name}_{chunk_count:04d}.wav")
            sf.write(out_path, full_segment, target_sr)
            chunk_count += 1
        
        # FINAL CHECK: If VAD produced 0 chunks (maybe all were too short?), force fixed slicing
        if chunk_count == 0:
            print("Warning: VAD processing resulted in 0 chunks (segments likely too short). Fallback to fixed slicing.")
            use_fixed_slicing = True

    if use_fixed_slicing:
        # Simple fixed-length slicing (10s chunks with 1s overlap)
        chunk_len = 10 * target_sr
        hop_len = 9 * target_sr
        for start in range(0, len(y) - chunk_len, hop_len):
            segment = y[start : start + chunk_len]
            out_path = os.path.join(output_dir, f"{base_name}_{chunk_count:04d}.wav")
            sf.write(out_path, segment, target_sr)
            chunk_count += 1
            
        # Handle the remainder if it's long enough
        last_start = (len(y) // hop_len) * hop_len
        if len(y) - last_start >= min_len * target_sr:
             segment = y[last_start:]
             out_path = os.path.join(output_dir, f"{base_name}_{chunk_count:04d}.wav")
             sf.write(out_path, segment, target_sr)
             chunk_count += 1
            
    else:
        # This block was moved into "if not use_fixed_slicing" above
        pass
            
    return chunk_count

def prepare_dataset():
    print("--- Preparing Oxxxymiron Rapper Dataset (Advanced Preprocessing) ---")
    
    source_file = "datasets/oxxxymiron_dataset_raw.wav"
    # New output directory
    target_dir = "datasets/rapper_finetune_preprocessed"
    
    if not os.path.exists(source_file):
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
    --run-name "rapper_oxxxy_finetune_v2" ^
    --batch-size 4 ^
    --max-epochs 100 ^
    --save-every 200 ^
    --num-workers 2

echo.
echo Training Finished!
echo Checkpoint: runs/rapper_oxxxy_finetune_v2/ft_model.pth
pause
"""
        # Save bat file with a new name to distinguish
        batch_file = os.path.join(os.path.dirname(__file__), "train_rapper_preprocessed.bat")
        with open(batch_file, "w") as f:
            f.write(batch_content)
        print(f"Created training script: {batch_file}")
        
    except Exception as e:
        print(f"Error generating script: {e}")

if __name__ == "__main__":
    prepare_dataset()
