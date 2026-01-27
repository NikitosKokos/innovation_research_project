import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from config import Config
from model import SeedVCModel
from stream import VoiceConversionStream

def main():
    print("==================================================")
    print("   Seed-VC Real-Time Voice Conversion (Jetson)    ")
    print("==================================================")
    
    # 1. Load Configuration
    config = Config()
    use_optimized = getattr(config, 'USE_OPTIMIZED_MODEL', False)
    
    if use_optimized:
        print(f"\n[Config] Using Optimized Model: {config.CHECKPOINT_PATH}")
    else:
        print(f"\n[Config] Using Original Model: {config.CHECKPOINT_PATH}")
    
    print(f"[Config] Target Voice:    {config.TARGET_VOICE_PATH}")
    print(f"[Config] Block Size:      {config.BLOCK_SIZE} samples (~{config.BLOCK_SIZE/config.SAMPLE_RATE*1000:.1f}ms)")
    print(f"[Config] Diffusion Steps: {config.DIFFUSION_STEPS} (optimized for Jetson Nano)")
    print(f"[Config] CFG Rate:         {config.INFERENCE_CFG_RATE} (0.0 = faster, 0.7 = better quality)")
    print(f"[Config] Device:          {config.DEVICE}")
    print(f"[Config] FP16:            {config.FP16} (using autocast for safety)")
    print(f"[Config] TensorRT:        {getattr(config, 'USE_TENSORRT', False)}")
    print(f"\n[Info] Optimizations for Jetson Nano:")
    print(f"       - Diffusion steps: {config.DIFFUSION_STEPS} (6 = best quality, 3 = faster)")
    print(f"       - CFG rate: {config.INFERENCE_CFG_RATE} (0.3 = good quality, 0.7 = best)")
    print(f"       - Chunks to accumulate: {getattr(config, 'CHUNKS_TO_ACCUMULATE', 6)} (reduces lag)")
    print(f"       - Block size: {config.BLOCK_SIZE} samples (~{config.BLOCK_SIZE/config.SAMPLE_RATE*1000:.1f}ms @ {config.SAMPLE_RATE}Hz)")
    print(f"       - Whisper on GPU (faster processing, falls back to CPU if needed)")
    print(f"       - CAM++ on GPU (faster processing, falls back to CPU if needed)")
    print(f"       - FP16: {'ENABLED' if config.FP16 else 'DISABLED'} (disabled by default to prevent memory issues)")
    print(f"       - Quantization: {'ENABLED' if getattr(config, 'ENABLE_QUANTIZATION', False) else 'DISABLED'} ({getattr(config, 'QUANTIZATION_TYPE', 'none')})")
    print(f"       - Output gain: {getattr(config, 'OUTPUT_GAIN', 1.0)}x")
    print(f"       - Lite preprocessing: DISABLED")
    print(f"       - Queue size limited to prevent memory buildup")
    if getattr(config, 'PASSTHROUGH_MODE', False):
        print(f"       - ⚠️  PASSTHROUGH MODE: Audio I/O test (no voice conversion)")
    if getattr(config, 'USE_TENSORRT', False):
        print(f"       - TensorRT: ENABLED (faster inference)")
    else:
        print(f"       - TensorRT: DISABLED (see TENSORRT_SETUP.md to enable)")
    
    # Check if model exists
    if not os.path.exists(config.CHECKPOINT_PATH):
        print(f"\n[Error] Model not found: {config.CHECKPOINT_PATH}")
        if use_optimized:
            print("Please run edge optimization first:")
            print("  python project/edge_optimization/find_optimal_config.py")
            print("\nOr set USE_OPTIMIZED_MODEL = False in config.py to use the original model")
        return
    
    if not os.path.exists(config.TARGET_VOICE_PATH):
        print(f"\n[Error] Target voice file not found: {config.TARGET_VOICE_PATH}")
        print("Please check config.py and set a valid TARGET_VOICE_PATH.")
        return

    # 2. Initialize Model
    print("\n[Init] Initializing AI Model (this may take a minute)...")
    try:
        model = SeedVCModel(config)
        
        # Pre-calculate target style
        model.set_target(config.TARGET_VOICE_PATH)
        
    except Exception as e:
        print(f"\n[Error] Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Initialize Stream
    print("\n[Init] Setting up Audio Stream...")
    vc_stream = VoiceConversionStream(config, model)

    # 4. Main Loop
    print("\n==================================================")
    print("   Ready to Start!                                ")
    print("==================================================")
    print("Commands:")
    print("  [s] Start Conversion")
    print("  [q] Quit")
    
    while True:
        cmd = input("\nCommand > ").lower().strip()
        
        if cmd == 's':
            if not vc_stream.running:
                vc_stream.start()
                print("Streaming started. Press 'Enter' to stop...")
                input() # Wait for Enter
                vc_stream.stop()
            else:
                print("Stream is already running.")
                
        elif cmd == 'q':
            print("Exiting...")
            if vc_stream.running:
                vc_stream.stop()
            break
        
        else:
            print("Unknown command.")

if __name__ == "__main__":
    main()
