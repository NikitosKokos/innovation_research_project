import os

class Config:
    # --- Paths ---
    # Root directory of the project (assuming this script is run from project root or adjusted)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    
    # Model Paths
    MODEL_DIR = os.path.join(PROJECT_ROOT, "AI_models", "russian_finetune_small_v3")
    CHECKPOINT_PATH = os.path.join(MODEL_DIR, "ft_model.pth")
    CONFIG_PATH = os.path.join(MODEL_DIR, "config_dit_mel_seed_uvit_whisper_small_wavenet.yml")
    
    # Target Voice Reference (The voice you want to sound like)
    # You can change this to any wav file
    TARGET_VOICE_PATH = os.path.join(PROJECT_ROOT, "examples", "reference", "s1p1.wav") 

    # --- Audio Settings ---
    SAMPLE_RATE = 44100  # Standard sample rate
    
    # Block Size / Chunk Size
    # For Real-Time (<1s latency): Try 4096, 8192, or 16384 (samples)
    # Note: Seed-VC is heavy. Too small blocksize (e.g. 512) might cause processing to lag behind.
    # 16384 samples @ 44.1kHz is ~0.37 seconds. This is a good balance for "near real-time".
    BLOCK_SIZE = 16384 
    
    # Audio Devices (Use 'pulse' or 'default' for system default, or specific IDs)
    INPUT_DEVICE = "pulse" 
    OUTPUT_DEVICE = "pulse"

    # --- Inference Settings ---
    # Number of diffusion steps. Lower = Faster but lower quality.
    # 10 is standard. 1-4 is very fast but robotic.
    DIFFUSION_STEPS = 5 
    
    # Length adjustment (1.0 = same speed as input)
    LENGTH_ADJUST = 1.0
    
    # Inference CFG rate (0.7 is standard)
    INFERENCE_CFG_RATE = 0.7
    
    # Pitch Shift (Semitones)
    PITCH_SHIFT = 0
    
    # F0 Condition (True/False) - Use False for the small model usually
    F0_CONDITION = False
    
    # Auto F0 Adjustment
    AUTO_F0_ADJUST = False

    # --- System ---
    # Force CPU if needed, otherwise uses CUDA if available
    DEVICE = "cuda" # 'cuda' or 'cpu'
    
    # Floating point precision
    FP16 = True
