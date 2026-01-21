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
    print(f"\n[Config] Target Voice: {config.TARGET_VOICE_PATH}")
    print(f"[Config] Block Size:   {config.BLOCK_SIZE}")
    print(f"[Config] Device:       {config.DEVICE}")
    
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
