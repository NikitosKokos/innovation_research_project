import os

class Config:
    # --- Paths ---
    # Root directory of the project (assuming this script is run from project root or adjusted)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    
    # Model Paths
    # Set USE_OPTIMIZED_MODEL = False to use the original stable model (recommended for quality)
    USE_OPTIMIZED_MODEL = False  # Set to False to avoid shape mismatch issues
    
    if USE_OPTIMIZED_MODEL:
        # Using optimized edge model from edge_optimization
        MODEL_DIR = os.path.join(PROJECT_ROOT, "AI_models", "best")
        CHECKPOINT_PATH = os.path.join(MODEL_DIR, "Prune0%_FP16_BT0.1_ECL1.0_ECR0.02_DS6.pth")
    else:
        # Using original fine-tuned model (more stable)
        MODEL_DIR = os.path.join(PROJECT_ROOT, "AI_models", "rapper_oxxxy_finetune")
        CHECKPOINT_PATH = os.path.join(MODEL_DIR, "ft_model.pth")
    
    # Config path remains the same (optimized model uses same config structure)
    CONFIG_PATH = os.path.join(PROJECT_ROOT, "AI_models", "rapper_oxxxy_finetune", "config_dit_mel_seed_uvit_whisper_small_wavenet.yml")
    
    # Target Voice Reference (The voice you want to sound like)
    # You can change this to any wav file
    TARGET_VOICE_PATH = os.path.join(PROJECT_ROOT, "audio_inputs", "reference", "ref01_processed.wav") 

    # --- Audio Settings ---
    # Model expects 22050 Hz (from config_dit_mel_seed_uvit_whisper_small_wavenet.yml)
    # Using 22050 to avoid slow resampling (was 44100 which required resampling taking 16-20 seconds!)
    SAMPLE_RATE = 22050  # Matches model's expected sample rate
    
    # Block Size / Chunk Size
    # Smaller chunks = faster processing, lower latency
    # 512 samples @ 22kHz = ~0.023s - very fast processing, lower latency
    # Smaller chunks process faster and reduce queue buildup
    BLOCK_SIZE = 512  # ~0.023s @ 22kHz, optimized for speed and lower latency 
    
    # Audio Devices (Use 'pulse' or 'default' for system default, or specific IDs)
    INPUT_DEVICE = "pulse" 
    OUTPUT_DEVICE = "pulse"

    # --- Inference Settings ---
    # Balance between quality and speed
    # More steps = better quality (smoother, less noise), but slower
    # Reduced to 4 for aggressive latency optimization (User request: "1-2s latency")
    DIFFUSION_STEPS = 6  
    
    # Length adjustment (1.0 = same speed as input)
    LENGTH_ADJUST = 1.0
    
    # Inference CFG rate
    # 0.7 = high quality (recommended by official GUI)
    # 0.0 = fastest, but poor quality
    # Changed to 0.0 for "near real-time" latency optimization (Step 3)
    INFERENCE_CFG_RATE = 0.0  
    
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
    # On Jetson Nano, FP16 is RECOMMENDED - Jetson has hardware FP16 acceleration
    # FP16 provides 2x speedup and 2x memory reduction on Jetson Nano
    FP16 = True  # ENABLED for Jetson Nano - uses hardware FP16 acceleration
    
    # --- Edge Optimization ---
    # Using pre-optimized model from edge_optimization (FP16 quantized)
    # The model is already optimized, so we don't need to apply additional optimization
    ENABLE_EDGE_OPTIMIZATION = False  # Enabled for pruning
    PRUNING_RATIO = 0.0  # Aggressive 50% pruning to speed up inference
    
    # --- Jetson Nano Optimizations ---
    # Layer Skipping (Depth Reduction)
    # Skip layers during inference to speed up processing
    # Example: [8, 9, 10, 11] will skip layers 8, 9, 10, 11
    # Empty list [] means no skipping
    # Aggressive skipping: Middle 5 layers (4, 5, 6, 7, 8)
    SKIP_LAYERS = [6]  # Skip last 5 layers (out of 13) for testing

    # Chunks to accumulate before processing (model needs sufficient context)
    # 43 chunks (512*43 = 22016 samples) = ~1.0s @ 22kHz
    # Reduced to ~1.0s (43 chunks) for "near real-time" latency optimization (Step 4)
    CHUNKS_TO_ACCUMULATE = 43
    
    # Disable progress bars for real-time performance (reduces I/O overhead)
    DISABLE_PROGRESS_BARS = True
    
    # Enable CUDA optimizations
    TORCH_CUDA_OPTIMIZATIONS = True  # Enable PyTorch CUDA optimizations
    
    # Enable Torch Compile (PyTorch 2.0+)
    # Can significantly speed up inference on Jetson Orin Nano (Ampere architecture)
    # Note: First inference will be slow due to compilation
    USE_TORCH_COMPILE = True
    
    # --- Vocoder Settings ---
    # 'bigvgan' = Best quality, but heavy (100M+ params). Slow on Jetson.
    # 'hifigan' = Much faster (~13M params), good quality. Recommended for Jetson.
    VOCODER_TYPE = 'bigvgan'
    
    # --- TensorRT Settings ---
    # Path to TensorRT engine (if available, will use TensorRT instead of PyTorch)
    # Set to None to use PyTorch, or path to .trt file to use TensorRT
    TENSORRT_ENGINE_PATH = os.path.join(MODEL_DIR, "model_fp16.trt")
    USE_TENSORRT = False  # Set to True to enable TensorRT inference
    # Note: You must generate the engine first using convert_to_tensorrt.py
    
    # --- Quantization Settings ---
    # DISABLED: Dynamic quantization fails with weight_norm layers
    # FP16 is better for Jetson Nano (hardware acceleration)
    # TensorRT would be ideal but requires ONNX conversion
    ENABLE_QUANTIZATION = False  # DISABLED - FP16 is better for Jetson Nano
    QUANTIZATION_TYPE = "dynamic"  # Not used when ENABLE_QUANTIZATION = False
    
    # --- Debug/Test Settings ---
    # Enable pass-through mode to test audio I/O without model processing
    # Set to True to just echo input to output (useful for testing audio pipeline)
    PASSTHROUGH_MODE = False  # Set to True to test audio I/O without model
    
    # Enable test tone to verify audio output is working
    # Set to True to play a test tone instead of processed audio (for debugging)
    TEST_TONE_MODE = False  # Set to True to play test tone (440Hz sine wave)
    
    # Output gain/amplification (to boost quiet model output)
    # Lowered to 3.0x to prevent clipping noise
    OUTPUT_GAIN = 3.0  
    # Note: Values > 5.0 may cause clipping/distortion and buzzing sounds
    # If too quiet, increase gradually (3.5, 4.0, etc.) but not above 5.0