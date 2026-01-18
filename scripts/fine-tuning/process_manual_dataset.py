import os
import glob
from pydub import AudioSegment
from tqdm import tqdm

def process_manual_dataset(source_dir="datasets/ru-RU", output_dir="datasets/russian_finetune", count=200):
    print(f"Processing audio from {source_dir} to {output_dir}...")
    
    # Check if source exists, if not try just ru-RU (backwards compat)
    if not os.path.exists(source_dir):
        if os.path.exists("ru-RU"):
            source_dir = "ru-RU"
            print(f"Redirecting to {source_dir}...")
        else:
            print(f"Error: Source directory {source_dir} not found.")
            return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Find files (recursive search for webm or mp3)
    # Dmitri 1.0 uses webm. Common Voice uses mp3.
    files = []
    for ext in ['*.webm', '*.mp3', '*.ogg', '*.wav']:
        files.extend(glob.glob(os.path.join(source_dir, '**', ext), recursive=True))
        
    if not files:
        print(f"Error: No audio files found in {source_dir}. Please check the path.")
        return

    print(f"Found {len(files)} files. Converting {min(len(files), count)}...")
    
    processed_count = 0
    pbar = tqdm(total=min(len(files), count))
    
    for filepath in files:
        if processed_count >= count:
            break
            
        try:
            # Load audio
            # Pydub handles webm if ffmpeg is installed/available
            audio = AudioSegment.from_file(filepath)
            
            # Convert to mono and set frame rate (optional but good practice)
            # The training script handles resampling, but converting to wav is key.
            audio = audio.set_channels(1) 
            
            # Create output filename
            basename = os.path.basename(filepath)
            filename = os.path.splitext(basename)[0] + ".wav"
            output_path = os.path.join(output_dir, filename)
            
            # Export as WAV
            audio.export(output_path, format="wav")
            
            processed_count += 1
            pbar.update(1)
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
            
    pbar.close()
    print(f"\nSuccessfully converted {processed_count} files to {output_dir}")
    print("You can now run scripts/run_manual_finetune.bat")

if __name__ == "__main__":
    # Ensure pydub is installed (it was in requirements)
    # User needs ffmpeg installed on system for pydub to read webm
    process_manual_dataset()
