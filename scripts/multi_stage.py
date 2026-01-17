#!/usr/bin/env python3
"""
Multi-Stage Voice Conversion for Maximum Quality
"""

"""
Multi-Stage Voice Conversion for Maximum Quality
"""

import librosa
import soundfile as sf
import numpy as np
from scipy import signal
import subprocess
import os
import sys
from pathlib import Path

# Get the script's directory to find inference.py
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
INFERENCE_SCRIPT = PROJECT_ROOT / "inference.py"

def stage1_preprocess(input_audio, output_path):
    """
    Stage 1: Aggressive preprocessing
    """
    audio, sr = librosa.load(input_audio, sr=22050)
    
    # 1. Spectral gating (remove background noise)
    try:
        import noisereduce as nr
        audio = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.8)
    except ImportError:
        print("⚠️  noisereduce not installed, skipping noise reduction. Install with: pip install noisereduce")
        # Continue without noise reduction
    
    # 2. Remove DC offset
    audio = audio - np.mean(audio)
    
    # 3. High-pass filter (remove rumble <80Hz)
    sos = signal.butter(4, 80, 'highpass', fs=sr, output='sos')
    audio = signal.sosfilt(sos, audio)
    
    # 4. De-essing (reduce harsh sibilance)
    # Target 6-10kHz range
    sos_deess = signal.butter(2, [6000, 10000], 'bandstop', fs=sr, output='sos')
    sibilance = signal.sosfilt(sos_deess, audio)
    audio = audio - 0.3 * sibilance  # Reduce by 30%
    
    # 5. Gentle compression (even dynamics)
    audio = np.tanh(audio * 1.2)  # Soft clipping
    
    # 6. Normalize to -3dB
    audio = librosa.util.normalize(audio) * 0.7
    
    sf.write(output_path, audio, sr)
    return output_path


def stage2_seed_vc_conversion(preprocessed_audio, reference, output_dir):
    """
    Stage 2: Seed-VC conversion with optimal settings
    """
    # Use sys.executable to use the current Python (venv)
    cmd = [
        sys.executable, str(INFERENCE_SCRIPT),
        "--source", preprocessed_audio,
        "--target", reference,
        "--output", output_dir,
        "--diffusion-steps", "40",
        "--inference-cfg-rate", "0.65",
        "--f0-condition", "True",
        "--auto-f0-adjust", "True"
    ]
    
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    
    # Find output file
    output_file = None
    for file in os.listdir(output_dir):
        if file.endswith('.wav'):
            output_file = os.path.join(output_dir, file)
            break
    
    return output_file


def stage3_postprocess(converted_audio, original_audio, output_path, blend_ratio=0.15):
    """
    Stage 3: Post-processing for naturalness
    """
    converted, sr = librosa.load(converted_audio, sr=22050)
    original, _ = librosa.load(original_audio, sr=22050)
    
    # Ensure same length
    min_len = min(len(converted), len(original))
    converted = converted[:min_len]
    original = original[:min_len]
    
    # 1. Blend with original for naturalness (preserve some original characteristics)
    blended = (1 - blend_ratio) * converted + blend_ratio * original
    
    # 2. Match original's spectral envelope (preserves pronunciation clarity)
    # Extract formants from original, apply to converted
    original_stft = librosa.stft(original)
    converted_stft = librosa.stft(blended)
    
    # Spectral envelope matching
    original_mag = np.abs(original_stft)
    converted_mag = np.abs(converted_stft)
    
    # Smooth spectral envelope
    original_envelope = signal.medfilt(original_mag, kernel_size=[1, 11])
    converted_envelope = signal.medfilt(converted_mag, kernel_size=[1, 11])
    
    # Apply original envelope characteristics (30% blend)
    target_envelope = 0.7 * converted_envelope + 0.3 * original_envelope
    ratio = target_envelope / (converted_envelope + 1e-10)
    
    enhanced_stft = converted_stft * ratio
    enhanced = librosa.istft(enhanced_stft)
    
    # 3. Subtle reverb (adds warmth, reduces artifacts)
    # Create impulse response for small room
    reverb_ir = np.zeros(int(0.05 * sr))  # 50ms reverb
    reverb_ir[0] = 1.0
    for i in range(1, len(reverb_ir)):
        reverb_ir[i] = reverb_ir[i-1] * 0.95 * np.random.randn() * 0.1
    
    enhanced_with_reverb = signal.convolve(enhanced, reverb_ir, mode='same')
    enhanced = 0.95 * enhanced + 0.05 * enhanced_with_reverb
    
    # 4. De-click (remove processing artifacts)
    enhanced = signal.medfilt(enhanced, kernel_size=3)
    
    # 5. Final normalization
    enhanced = librosa.util.normalize(enhanced) * 0.8
    
    sf.write(output_path, enhanced, sr)
    return output_path


def multi_stage_conversion(input_audio, reference_audio, output_path):
    """
    Complete multi-stage pipeline
    """
    print("🔄 Stage 1: Preprocessing...")
    temp_dir = "temp_processing"
    os.makedirs(temp_dir, exist_ok=True)
    
    preprocessed = os.path.join(temp_dir, "preprocessed.wav")
    stage1_preprocess(input_audio, preprocessed)
    
    print("🔄 Stage 2: Voice Conversion...")
    conversion_dir = os.path.join(temp_dir, "converted")
    os.makedirs(conversion_dir, exist_ok=True)
    converted = stage2_seed_vc_conversion(preprocessed, reference_audio, conversion_dir)
    
    print("🔄 Stage 3: Post-processing...")
    final = stage3_postprocess(converted, input_audio, output_path)
    
    print(f"✅ Complete! Output: {output_path}")
    return final


# Usage
if __name__ == "__main__":
    multi_stage_conversion(
        "audio_inputs/user/input.wav",
        "audio_inputs/reference/ref2.wav",
        "audio_outputs/final_enhanced.wav"
    )
