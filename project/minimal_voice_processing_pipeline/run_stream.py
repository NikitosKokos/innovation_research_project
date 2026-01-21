import sounddevice as sd
import numpy as np
import sys
import time
import json
import os
from datetime import datetime

def list_devices():
    """Lists available audio devices, filtering out internal Jetson APE devices."""
    print("\n--- Available Audio Devices ---")
    devices = sd.query_devices()
    valid_devices = []
    
    for i, device in enumerate(devices):
        # Filter out internal Jetson APE devices to reduce clutter
        if "APE" in device['name']:
            continue
            
        io_str = []
        if device['max_input_channels'] > 0:
            io_str.append(f"In: {device['max_input_channels']}")
        if device['max_output_channels'] > 0:
            io_str.append(f"Out: {device['max_output_channels']}")
        
        # Only show devices that have at least some I/O
        if io_str:
            io_display = ", ".join(io_str)
            print(f"ID {i}: {device['name']} ({io_display}) - Host API: {device['hostapi']}")
            valid_devices.append(i)
            
    return devices, valid_devices

def get_device_selection(prompt, valid_indices, devices, is_input=True):
    """Asks user for a device index and validates channels."""
    while True:
        try:
            selection = input(f"\n{prompt}: ")
            idx = int(selection)
            if idx in valid_indices:
                # Validation: Check if device actually has the required channels
                device = devices[idx]
                if is_input and device['max_input_channels'] == 0:
                    print(f"Error: Device '{device['name']}' has 0 Input channels. Please select another.")
                    continue
                if not is_input and device['max_output_channels'] == 0:
                    print(f"Error: Device '{device['name']}' has 0 Output channels. Please select another.")
                    continue
                return idx
            print(f"Invalid ID. Please choose one of: {valid_indices}")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Global list to store latency measurements
latency_measurements = []

def audio_callback(indata, outdata, frames, time_info, status):
    """Callback for audio processing."""
    if status:
        # Don't print to stderr during tuning to keep output clean, 
        # but you can log it if needed.
        pass
    
    # Pass-through: Copy input to output
    outdata[:] = indata
    
    # Calculate latency for this specific block
    try:
        current_latency = time_info.outputBufferDacTime - time_info.inputBufferAdcTime
        latency_measurements.append(current_latency)
    except:
        pass

def save_metrics(metrics, blocksize):
    """Saves metrics to a JSON file with auto-incrementing filename."""
    output_dir = "project/minimal_voice_processing_pipeline/metrics"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Find the next available index for this blocksize
    idx = 1
    while True:
        filename = f"metrics_{blocksize}_blocksize_{idx:02d}.json"
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            break
        idx += 1
        
    try:
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"\n[Success] Metrics saved to: {filepath}")
    except Exception as e:
        print(f"\n[Error] Failed to save metrics: {e}")

def run_stream(input_id, output_id, samplerate, blocksize, input_name, output_name):
    """Runs the audio stream for a specific configuration."""
    latency_measurements.clear()
    
    print(f"\n--- Starting Stream (Block Size: {blocksize}) ---")
    print(f"Buffer Duration: {blocksize/samplerate*1000:.2f} ms")
    print("Press Enter to stop...")
    
    start_time = time.time()
    
    try:
        with sd.Stream(device=(input_id, output_id),
                       samplerate=samplerate, blocksize=blocksize,
                       channels=1, callback=audio_callback) as stream:
            input() # Wait for Enter
            
    except Exception as e:
        print(f"Error: {e}")
        return

    end_time = time.time()
    duration = end_time - start_time

    # Process and Save Metrics
    if latency_measurements:
        # Filter out the first few frames which might be unstable
        valid_measurements = latency_measurements[5:] if len(latency_measurements) > 5 else latency_measurements
        
        if valid_measurements:
            avg_latency = float(np.mean(valid_measurements) * 1000)
            min_latency = float(np.min(valid_measurements) * 1000)
            max_latency = float(np.max(valid_measurements) * 1000)
            jitter = float(np.std(valid_measurements) * 1000)
            
            print("\n=== Final Performance Report ===")
            print(f"  Block Size:                {blocksize} frames")
            print(f"  Buffer Duration:           {blocksize/samplerate*1000:.2f} ms")
            print(f"  Average Roundtrip Latency: {avg_latency:.2f} ms")
            print(f"  Jitter (Stability):        {jitter:.2f} ms")
            
            # Prepare data for JSON
            metrics_data = {
                "timestamp": datetime.now().isoformat(),
                "configuration": {
                    "input_device": input_name,
                    "output_device": output_name,
                    "samplerate": samplerate,
                    "blocksize": blocksize,
                    "channels": 1
                },
                "performance": {
                    "avg_latency_ms": round(avg_latency, 2),
                    "min_latency_ms": round(min_latency, 2),
                    "max_latency_ms": round(max_latency, 2),
                    "jitter_ms": round(jitter, 2),
                    "buffer_duration_ms": round(blocksize/samplerate*1000, 2),
                    "total_blocks_processed": len(latency_measurements),
                    "test_duration_seconds": round(duration, 2)
                },
                "raw_measurements_sample": [round(x*1000, 2) for x in valid_measurements[:10]] # First 10 samples
            }
            
            save_metrics(metrics_data, blocksize)
            
            print("\n[Analysis]")
            print(f"  Your latency is {avg_latency:.0f}ms.")
        else:
            print("\nNot enough data collected.")
    else:
        print("\nNo latency data collected.")
    print("==== Script ended successfully ====")

def main():
    print("=== Basic Voice Processing Pipeline Test (Pass-through) ===")
    print("This tool tests audio input-to-output latency and connectivity on your Jetson Nano.")

    # 1. List and Select Devices
    all_devices, valid_indices = list_devices()
    
    print("\n[Device Selection Guide]")
    print("1. RECOMMENDED: Select 'default' or 'pulse' for BOTH input and output.")
    print("   (This uses your Ubuntu Sound Settings to route audio)")
    
    input_device_id = get_device_selection(
        "Enter ID for INPUT device (e.g., 'default' or Mic)", 
        valid_indices, 
        all_devices, 
        is_input=True
    )
    
    output_device_id = get_device_selection(
        "Enter ID for OUTPUT device (e.g., 'default' or HDMI)", 
        valid_indices, 
        all_devices, 
        is_input=False
    )

    # 2. Mode Selection
    print("\n[Select Mode]")
    print("1. Standard Test (Safe, Higher Latency - Blocksize 4096)")
    print("2. Low Latency Test (Aggressive - Blocksize 512)")
    print("3. Custom Blocksize")
    
    mode = input("Enter mode (1-3): ")
    
    samplerate = 44100
    if mode == '2':
        blocksize = 512
    elif mode == '3':
        try:
            blocksize = int(input("Enter blocksize (e.g., 128, 256, 1024): "))
        except:
            blocksize = 4096
    else:
        blocksize = 4096

    # 3. Run Stream
    run_stream(
        input_device_id, 
        output_device_id, 
        samplerate, 
        blocksize,
        all_devices[input_device_id]['name'],
        all_devices[output_device_id]['name']
    )

if __name__ == "__main__":
    main()
