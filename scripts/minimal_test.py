#!/usr/bin/env python3
"""
Minimal High-Quality Voice Conversion
Start simple, then add complexity
"""

import subprocess
import sys
from pathlib import Path
import os

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
INFERENCE_SCRIPT = PROJECT_ROOT / "inference.py"

def minimal_quality_conversion(source, reference, output_dir):
    """
    Simplest high-quality conversion - no fancy processing
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Start with MINIMAL settings
    cmd = [
        sys.executable, str(INFERENCE_SCRIPT),
        "--source", source,
        "--target", reference,
        "--output", output_dir,
        "--diffusion-steps", "25",  # Lower first
        "--inference-cfg-rate", "0.5",  # Much lower for naturalness
        # Remove F0 conditioning temporarily to test
    ]
    
    print("🔄 Testing minimal conversion...")
    print(f"Settings: diffusion=25, cfg=0.5, no F0 conditioning")
    
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    
    # Find output
    for file in os.listdir(output_dir):
        if file.endswith('.wav'):
            return os.path.join(output_dir, file)

if __name__ == "__main__":
    result = minimal_quality_conversion(
        "audio_inputs/user/input.wav",
        "audio_inputs/reference/ref.wav",
        "audio_outputs/minimal_test"
    )
    print(f"\n✅ Test complete: {result}")
    print("\nListen to this output:")
    print("- If STILL robotic → Problem is reference audio")
    print("- If better → Problem was post-processing")
    print("- If worse → Need to adjust parameters")
