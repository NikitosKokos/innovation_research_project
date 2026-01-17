#!/usr/bin/env python3
"""
Reference Audio Quality Checker
"""

import librosa
import numpy as np
import soundfile as sf

def check_reference_quality(reference_path):
    """
    Analyze reference audio for quality issues
    """
    audio, sr = librosa.load(reference_path, sr=22050)
    
    print(f"\n{'='*60}")
    print("📊 REFERENCE AUDIO QUALITY ANALYSIS")
    print(f"{'='*60}\n")
    
    # 1. Duration check
    duration = len(audio) / sr
    print(f"Duration: {duration:.2f} seconds")
    if duration < 10:
        print("  ⚠️  WARNING: Too short! Need 10-30 seconds")
        print("  → Solution: Use longer clip")
    elif duration > 40:
        print("  ⚠️  WARNING: Too long! May cause issues")
        print("  → Solution: Trim to 15-25 seconds")
    else:
        print("  ✅ Good duration")
    
    # 2. Volume check
    rms = np.sqrt(np.mean(audio**2))
    print(f"\nRMS Energy: {rms:.4f}")
    if rms < 0.05:
        print("  ⚠️  WARNING: Too quiet!")
        print("  → Solution: Normalize audio")
    elif rms > 0.5:
        print("  ⚠️  WARNING: Too loud (clipping risk)")
        print("  → Solution: Reduce volume")
    else:
        print("  ✅ Good volume level")
    
    # 3. Noise level check
    # Estimate noise floor
    noise_threshold = np.percentile(np.abs(audio), 10)
    print(f"\nNoise Floor: {noise_threshold:.4f}")
    if noise_threshold > 0.02:
        print("  ⚠️  WARNING: High background noise!")
        print("  → Solution: Use cleaner audio or apply noise reduction")
    else:
        print("  ✅ Low noise")
    
    # 4. Frequency content
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    avg_centroid = np.mean(spectral_centroid)
    print(f"\nSpectral Centroid: {avg_centroid:.0f} Hz")
    if avg_centroid < 1000:
        print("  ⚠️  WARNING: Very dark/muffled sound")
    elif avg_centroid > 4000:
        print("  ⚠️  WARNING: Very bright/harsh sound")
    else:
        print("  ✅ Good tonal balance")
    
    # 5. Check for music/reverb
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    onset_density = np.sum(onset_env > 0.5) / len(onset_env)
    print(f"\nOnset Density: {onset_density:.3f}")
    if onset_density > 0.3:
        print("  ⚠️  WARNING: May contain music/multiple speakers")
        print("  → Solution: Extract clean vocal-only section")
    else:
        print("  ✅ Likely clean speech")
    
    # 6. Check for clipping
    clipping_samples = np.sum(np.abs(audio) > 0.99)
    clipping_percentage = (clipping_samples / len(audio)) * 100
    print(f"\nClipping: {clipping_percentage:.2f}%")
    if clipping_percentage > 0.1:
        print("  ⚠️  WARNING: Audio is clipping!")
        print("  → Solution: Use source with lower volume")
    else:
        print("  ✅ No clipping")
    
    # Final verdict
    print(f"\n{'='*60}")
    issues = []
    if duration < 10 or duration > 40:
        issues.append("duration")
    if rms < 0.05 or rms > 0.5:
        issues.append("volume")
    if noise_threshold > 0.02:
        issues.append("noise")
    if onset_density > 0.3:
        issues.append("music/reverb")
    if clipping_percentage > 0.1:
        issues.append("clipping")
    
    if not issues:
        print("✅ REFERENCE AUDIO IS GOOD QUALITY")
    else:
        print(f"⚠️  ISSUES FOUND: {', '.join(issues)}")
        print("\nThis may be causing robotic sound!")
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    check_reference_quality("audio_inputs/reference/ref2.wav")
