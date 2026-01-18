#!/usr/bin/env python3
"""
Custom Voice Conversion Script for Seed-VC
Uses the official inference.py from the repository

source ./.venv/Scripts/activate

python scripts/convert_voice.py   --source audio_inputs/user/input.wav   --reference audio_inputs/reference/ref.wav   --output audio_outputs/results

python scripts/convert_voice.py   --source audio_inputs/user/input.wav   --reference audio_inputs/reference/ref.wav   --output audio_outputs/results --f0-condition --auto-f0-adjust
"""

import os
import sys
import argparse
import subprocess
import warnings
from pathlib import Path

# Suppress warnings from subprocess
warnings.filterwarnings('ignore')

# Add the parent directory to path to import seed-vc modules
sys.path.insert(0, str(Path(__file__).parent.parent))

def preprocess_audio(audio_path, output_path):
    """
    Preprocess audio for better voice conversion quality
    Reduces noise and normalizes volume
    """
    import librosa
    import soundfile as sf
    import numpy as np
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=None)
    
    # Remove silence at start/end
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    # Normalize volume (prevent clipping and too-quiet audio)
    audio = librosa.util.normalize(audio)
    
    # Optional: light noise reduction
    # Uncomment if source has background noise
    # from scipy.signal import wiener
    # audio = wiener(audio)
    
    # Save preprocessed audio
    sf.write(output_path, audio, sr)
    
    return output_path


def convert_voice(
    source_audio: str,
    reference_audio: str,
    output_dir: str,
    diffusion_steps: int = 25,
    length_adjust: float = 1.0,
    inference_cfg_rate: float = 0.7,
    f0_condition: bool = False,
    auto_f0_adjust: bool = False,
    semi_tone_shift: int = 0,
    fp16: bool = True,
    preprocess: bool = True
):
    """
    Convert voice using Seed-VC
    
    Args:
        source_audio: Path to input audio file to convert
        reference_audio: Path to reference audio (target voice, 1-30 seconds)
        output_dir: Directory to save output
        diffusion_steps: Number of diffusion steps (25 default, 30-50 for singing)
        length_adjust: Speed adjustment (<1.0 faster, >1.0 slower)
        inference_cfg_rate: CFG rate for inference quality (0.0-1.0)
        f0_condition: Enable pitch conditioning (True for singing voice)
        auto_f0_adjust: Automatically adjust pitch
        semi_tone_shift: Pitch shift in semitones (for singing)
        fp16: Use FP16 for faster inference
    """
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if preprocess:
        print("📝 Preprocessing audio for better quality...")
        temp_dir = os.path.join(output_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        source_processed = os.path.join(temp_dir, "source_processed.wav")
        ref_processed = os.path.join(temp_dir, "ref_processed.wav")
        
        preprocess_audio(source_audio, source_processed)
        preprocess_audio(reference_audio, ref_processed)
        
        source_audio = source_processed
        reference_audio = ref_processed
    
    # Build command
    cmd = f"""python inference.py \
--source {source_audio} \
--target {reference_audio} \
--output {output_dir} \
--diffusion-steps {diffusion_steps} \
--length-adjust {length_adjust} \
--inference-cfg-rate {inference_cfg_rate}"""

    if f0_condition:
        cmd += " --f0-condition True"
    
    if auto_f0_adjust:
        cmd += " --auto-f0-adjust True"
    
    if semi_tone_shift != 0:
        cmd += f" --semi-tone-shift {semi_tone_shift}"
    
    if fp16:
        cmd += " --fp16 True"
    
    print(f"\n{'='*60}")
    print("SEED-VC Voice Conversion")
    print(f"{'='*60}")
    print(f"Source Audio:     {source_audio}")
    print(f"Reference Audio:  {reference_audio}")
    print(f"Output Directory: {output_dir}")
    print(f"Diffusion Steps:  {diffusion_steps}")
    print(f"Length Adjust:    {length_adjust}")
    print(f"F0 Condition:     {f0_condition}")
    print(f"Preprocessing:    {preprocess}")
    print(f"{'='*60}\n")
    
    # Execute conversion with better error handling
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✓ Conversion completed! Check output in: {output_dir}\n")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: Conversion failed with exit code {e.returncode}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⚠️  Conversion interrupted by user\n")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Seed-VC Voice Conversion Script")
    
    parser.add_argument(
        "--source",
        type=str,
        default="audio_inputs/user/input.wav",
        help="Path to source audio file"
    )
    
    parser.add_argument(
        "--reference",
        type=str,
        default="audio_inputs/reference/ref.wav",
        help="Path to reference audio file (target voice)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="audio_outputs/results",
        help="Output directory for converted audio"
    )
    
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=40,
        help="Number of diffusion steps (25-50, higher = better quality)"
    )
    
    parser.add_argument(
        "--length-adjust",
        type=float,
        default=1.0,
        help="Speed adjustment factor (<1.0 = faster, >1.0 = slower)"
    )
    
    parser.add_argument(
        "--inference-cfg-rate",
        type=float,
        default=0.8,
        help="Inference CFG rate (0.0-1.0)"
    )
    
    parser.add_argument(
        "--f0-condition",
        action="store_true",
        help="Enable F0 conditioning (use for singing voice conversion)"
    )
    
    parser.add_argument(
        "--auto-f0-adjust",
        action="store_true",
        help="Automatically adjust F0 (pitch)"
    )
    
    parser.add_argument(
        "--semi-tone-shift",
        type=int,
        default=0,
        help="Pitch shift in semitones (for singing voice)"
    )
    
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use FP16 precision for faster inference"
    )

    parser.add_argument(
        "--preprocess",
        action="store_true",
        default=True,
        help="Preprocess audio (noise reduction, normalization)"
    )
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not os.path.exists(args.source):
        print(f"❌ Error: Source audio not found: {args.source}")
        sys.exit(1)
    
    if not os.path.exists(args.reference):
        print(f"❌ Error: Reference audio not found: {args.reference}")
        sys.exit(1)
    
    # Run conversion
    convert_voice(
        source_audio=args.source,
        reference_audio=args.reference,
        output_dir=args.output,
        diffusion_steps=args.diffusion_steps,
        length_adjust=args.length_adjust,
        inference_cfg_rate=args.inference_cfg_rate,
        f0_condition=args.f0_condition,
        auto_f0_adjust=args.auto_f0_adjust,
        semi_tone_shift=args.semi_tone_shift,
        fp16=args.fp16,
        preprocess=args.preprocess
    )


if __name__ == "__main__":
    main()
