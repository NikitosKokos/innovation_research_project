#!/usr/bin/env python3
"""
Optimized Multi-Stage Voice Conversion
Fixed for robotic sound issues
"""

import librosa
import soundfile as sf
import numpy as np
from scipy import signal
import subprocess
import sys
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
INFERENCE_SCRIPT = PROJECT_ROOT / "inference.py"


def gentle_preprocess(input_audio, output_path):
    """
    GENTLE preprocessing - don't destroy naturalness
    """
    audio, sr = librosa.load(input_audio, sr=22050)
    
    # 1. Trim silence only
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    # 2. Gentle noise reduction (if available)
    try:
        import noisereduce as nr
        audio = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.5)  # 50% not 80%!
    except ImportError:
        pass
    
    # 3. Normalize only
    audio = librosa.util.normalize(audio) * 0.7
    
    sf.write(output_path, audio, sr)
    return output_path


def optimized_seed_vc(preprocessed_audio, reference, output_dir):
    """
    Seed-VC with anti-robotic settings
    """
    cmd = [
        sys.executable, str(INFERENCE_SCRIPT),
        "--source", preprocessed_audio,
        "--target", reference,
        "--output", output_dir,
        "--diffusion-steps", "40",  # Not too high
        "--inference-cfg-rate", "0.65",  # Lower for naturalness
        # Try WITHOUT F0 conditioning first
    ]
    
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    
    for file in os.listdir(output_dir):
        if file.endswith('.wav'):
            return os.path.join(output_dir, file)


def minimal_postprocess(converted_audio, original_audio, output_path):
    """
    MINIMAL post-processing - preserve what Seed-VC did
    """
    converted, sr = librosa.load(converted_audio, sr=22050)
    original, _ = librosa.load(original_audio, sr=22050)
    
    # Ensure same length
    min_len = min(len(converted), len(original))
    converted = converted[:min_len]
    original = original[:min_len]
    
    # Only blend 10% with original (less is more!)
    blended = 0.9 * converted + 0.1 * original
    
    # Light smoothing only
    blended = signal.savgol_filter(blended, window_length=5, polyorder=2)
    
    # Normalize
    blended = librosa.util.normalize(blended) * 0.75
    
    sf.write(output_path, blended, sr)
    return output_path


def optimized_conversion(input_audio, reference_audio, output_path):
    """
    Optimized pipeline - less is more
    """
    print("🔄 Stage 1: Gentle preprocessing...")
    temp_dir = "temp_optimized"
    os.makedirs(temp_dir, exist_ok=True)
    
    preprocessed = os.path.join(temp_dir, "preprocessed.wav")
    gentle_preprocess(input_audio, preprocessed)
    
    print("🔄 Stage 2: Optimized voice conversion...")
    conversion_dir = os.path.join(temp_dir, "converted")
    os.makedirs(conversion_dir, exist_ok=True)
    converted = optimized_seed_vc(preprocessed, reference_audio, conversion_dir)
    
    print("🔄 Stage 3: Minimal post-processing...")
    final = minimal_postprocess(converted, input_audio, output_path)
    
    print(f"✅ Complete! Output: {output_path}")
    return final


if __name__ == "__main__":
    optimized_conversion(
        "audio_inputs/user/input.wav",
        "audio_inputs/reference/ref.wav",
        "audio_outputs/optimized_output.wav"
    )
