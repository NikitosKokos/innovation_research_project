import os
import shutil
import soundfile as sf
import librosa
from datasets import load_dataset
from tqdm import tqdm

def prepare_dataset(output_dir="datasets/russian_finetune", count=200):
    print(f"Preparing Russian dataset with {count} samples from Common Voice 11.0...")
    
    # Ensure directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load Common Voice Russian (streaming)
    # Requires 'huggingface-cli login' to be done previously
    try:
        print("Loading dataset: mozilla-foundation/common_voice_11_0 (ru)...")
        dataset = load_dataset("mozilla-foundation/common_voice_11_0", "ru", split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        print("\nACTION REQUIRED:")
        print("1. Go to https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0 and accept terms.")
        print("2. Run 'huggingface-cli login' in your terminal and paste your Access Token.")
        return

    saved_count = 0
    iterator = iter(dataset)
    
    pbar = tqdm(total=count)
    
    while saved_count < count:
        try:
            item = next(iterator)
            
            # Access audio
            if 'audio' in item:
                audio_data = item['audio']
            else:
                continue

            audio_array = audio_data['array']
            sr = audio_data['sampling_rate']
            
            # Filter for reasonable length (1.5s to 15s)
            duration = len(audio_array) / sr
            if duration < 1.5 or duration > 15.0:
                continue
                
            # Create filename
            client_id = item.get('client_id', str(saved_count))[:12]
            # Sanitize
            client_id = "".join([c for c in client_id if c.isalnum()])
            
            filename = f"ru_cv_{saved_count:03d}_{client_id}.wav"
            filepath = os.path.join(output_dir, filename)
            
            # Save as WAV
            sf.write(filepath, audio_array, sr)
            
            saved_count += 1
            pbar.update(1)
            
        except StopIteration:
            print("End of dataset reached.")
            break
        except Exception as e:
            # print(f"Skipping: {e}")
            continue
            
    pbar.close()
    
    files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
    print(f"\nSuccessfully saved {len(files)} audio files to {output_dir}")

if __name__ == "__main__":
    prepare_dataset()
