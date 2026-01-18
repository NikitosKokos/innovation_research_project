import os
import glob
from pydub import AudioSegment
from tqdm import tqdm
import multiprocessing

def process_file(file_info):
    source_path, output_path, target_sr = file_info
    
    try:
        if os.path.exists(output_path):
            return "skipped" # Already exists

        # Load audio
        audio = AudioSegment.from_file(source_path)
        
        # Convert to mono and target sample rate (22050Hz for this model)
        audio = audio.set_channels(1).set_frame_rate(target_sr)
        
        # Filter: Skip if too short (< 1s) or too long (> 20s)
        if audio.duration_seconds < 1.0 or audio.duration_seconds > 20.0:
            return "filtered"

        # Export
        audio.export(output_path, format="wav")
        return "success"
    except Exception as e:
        print(f"Error processing {source_path}: {e}")
        return "error"

def main():
    source_dir = "datasets/ru-RU"
    output_dir = "datasets/russian_finetune"
    target_sr = 22050
    
    print(f"--- Preparing Russian Dataset ---")
    print(f"Source: {source_dir}")
    print(f"Target: {output_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Find all audio files (webm, mp3, wav, ogg)
    extensions = ['*.webm', '*.mp3', '*.wav', '*.ogg']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(source_dir, '**', ext), recursive=True))
    
    if not files:
        print(f"Error: No audio files found in {source_dir}")
        return

    print(f"Found {len(files)} candidate files.")
    
    # Prepare tasks for multiprocessing
    tasks = []
    for i, filepath in enumerate(files):
        # Create a unique filename to avoid collisions if folder structure is flat
        basename = os.path.basename(filepath)
        filename = f"ru_{i:04d}_{os.path.splitext(basename)[0]}.wav"
        output_path = os.path.join(output_dir, filename)
        tasks.append((filepath, output_path, target_sr))

    # Run conversion in parallel to speed up
    # Leave 2 cores free for system
    num_processes = max(1, multiprocessing.cpu_count() - 2)
    print(f"Processing with {num_processes} cores...")
    
    results = {"success": 0, "filtered": 0, "error": 0, "skipped": 0}
    
    with multiprocessing.Pool(num_processes) as pool:
        for res in tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks)):
            results[res] += 1
            
    print(f"\nProcessing Complete:")
    print(f"- Converted: {results['success']}")
    print(f"- Filtered (length): {results['filtered']}")
    print(f"- Errors: {results['error']}")
    print(f"- Skipped (existed): {results['skipped']}")
    print(f"\nReady for training in: {output_dir}")

if __name__ == "__main__":
    main()
