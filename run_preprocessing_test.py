import os
import sys
import numpy as np
import librosa
import soundfile as sf
import json

# Add root to path
sys.path.append(os.getcwd())

from project.preprocessing.pipeline import ReferenceVoiceProcessor, ReferenceProcessingConfig
from project.preprocessing.normalization import LUFSNormalizer
from project.preprocessing.denoising import get_denoiser

def measure_stats(audio, sr, name="Audio"):
    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))
    # Simple dynamic range estimate (peak vs RMS in dB)
    dr_db = 20 * np.log10(peak / (rms + 1e-10))
    
    print(f"--- {name} Stats ---")
    print(f"  RMS Level: {20 * np.log10(rms):.2f} dB")
    print(f"  Peak Level: {20 * np.log10(peak):.2f} dB")
    print(f"  Dynamic Range: {dr_db:.2f} dB")
    return rms, peak

def run_test():
    input_file = "audio_inputs/user/test01.mp3"
    output_file = "audio_outputs/test_preprocessing_demo.wav"
    
    if not os.path.exists(input_file):
        # Fallback if specific file doesn't exist
        inputs = [f for f in os.listdir("audio_inputs/user") if f.endswith(".wav") or f.endswith(".mp3")]
        if inputs:
            input_file = os.path.join("audio_inputs/user", inputs[0])
        else:
            print("No input file found to test.")
            return

    print(f"Testing preprocessing on: {input_file}")
    
    # Load Original
    y, sr = librosa.load(input_file, sr=22050)
    measure_stats(y, sr, "Original")
    
    # Setup Config
    config = ReferenceProcessingConfig(
        input_dir=os.path.dirname(input_file),
        output_dir="audio_outputs",
        sample_rate=22050,
        denoising_method="noisereduce", # Using the best available
        target_lufs=-23.0,
        compression_ratio=3.0
    )
    
    processor = ReferenceVoiceProcessor(config)
    
    # Process
    # We bypass the batch processor to run just one file explicitly for control
    print("\nRunning Pipeline...")
    
    # 1. Denoise
    print("1. Denoising (noisereduce)...")
    y_denoised = processor.denoiser.denoise(y)
    measure_stats(y_denoised, sr, "After Denoising")
    
    # 2. Normalize
    print("2. Normalizing (-23 LUFS)...")
    y_norm = processor.normalizer.normalize(y_denoised)
    measure_stats(y_norm, sr, "After Normalization")
    
    # 3. Compression
    print("3. Compressing (Ratio 3.0)...")
    y_final = processor.compressor.compress(y_norm)
    measure_stats(y_final, sr, "After Compression (Final)")
    
    # Save
    sf.write(output_file, y_final, sr)
    print(f"\nSaved processed file to: {output_file}")

if __name__ == "__main__":
    run_test()
