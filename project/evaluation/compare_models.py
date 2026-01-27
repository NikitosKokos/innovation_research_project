import os
import sys
import torch
import shutil
import librosa
import json
import matplotlib.pyplot as plt
import librosa.display
try:
    import torchaudio
    import torchaudio.transforms as T
    TORCHAUDIO_AVAILABLE = True
except (ImportError, OSError) as e:
    TORCHAUDIO_AVAILABLE = False
    torchaudio = None
    T = None
    print(f"Warning: torchaudio not available ({e}). Using librosa fallbacks.")
import numpy as np
import soundfile as sf

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from modules.campplus.DTDNN import CAMPPlus
from hf_utils import load_custom_model_from_hf
from project.preprocessing.evaluation import compute_pesq, compute_stoi
from project.preprocessing.denoising import get_denoiser
from project.preprocessing.normalization import LUFSNormalizer

# --- Metrics ---

def load_campplus(device):
    print("Loading CAM++ Speaker Encoder...")
    model = CAMPPlus(feat_dim=80, embedding_size=192)
    sd_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
    model.load_state_dict(torch.load(sd_path, map_location='cpu'))
    model.eval().to(device)
    return model

def get_embedding(path, model, device):
    # Load with librosa
    wav, sr = librosa.load(path, sr=None)
    wav = torch.from_numpy(wav).float()
    
    # Resample to 16k
    if sr != 16000:
        if TORCHAUDIO_AVAILABLE:
            wav = T.Resample(sr, 16000)(wav)
        else:
            # Fallback to librosa resampling
            wav_np = wav.numpy()
            wav_np = librosa.resample(wav_np, orig_sr=sr, target_sr=16000)
            wav = torch.from_numpy(wav_np).float()
            sr = 16000
        
    if wav.ndim == 1: wav = wav.unsqueeze(0)
    
    wav = wav.to(device)
    if TORCHAUDIO_AVAILABLE:
        feat = torchaudio.compliance.kaldi.fbank(wav, num_mel_bins=80, dither=0, sample_frequency=16000)
    else:
        # Fallback: use librosa mel spectrogram as approximation
        wav_np = wav.squeeze(0).cpu().numpy()
        feat = librosa.feature.melspectrogram(
            y=wav_np,
            sr=16000,
            n_mels=80,
            fmin=0,
            fmax=8000,
            n_fft=512,
            hop_length=160
        )
        # Convert to log scale and transpose to match kaldi format (time, freq)
        feat = torch.tensor(np.log(feat.T + 1e-10), dtype=torch.float32).to(device)
    feat = feat - feat.mean(dim=0, keepdim=True)
    
    with torch.no_grad():
        emb = model(feat.unsqueeze(0))
    return emb

def get_cosine_sim(emb1, emb2):
    return torch.nn.CosineSimilarity(dim=1)(emb1, emb2).item()

# --- Visualization Helper ---

def plot_mel_comparison(source_path, target_path, output_path, save_path, title_suffix=""):
    """
    Generates a 3-panel Mel Spectrogram comparison plot.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    paths = [source_path, target_path, output_path]
    titles = ["Source (Linguistic)", "Target (Timbre)", "Converted (Combined)"]
    
    for i, (p, t) in enumerate(zip(paths, titles)):
        y, sr = librosa.load(p, sr=22050)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        img = librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=axes[i], cmap='viridis')
        axes[i].set_title(f"{t} {title_suffix}")
        if i < 2: axes[i].set_xlabel('')
        fig.colorbar(img, ax=axes[i], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  -> Saved spectrogram comparison to: {save_path}")

# --- Preprocessing Helper ---

def preprocess_audio_for_vc(audio_path, output_path, target_sr=16000):
    """
    Cleans audio by denoising and normalizing to improve VC results and metrics.
    """
    wav, sr = librosa.load(audio_path, sr=target_sr)
    
    # Denoise
    denoiser = get_denoiser("noisereduce", sr=target_sr)
    wav = denoiser.denoise(wav)
    
    # Normalize LUFS
    normalizer = LUFSNormalizer(target_lufs=-23.0, sr=target_sr)
    wav = normalizer.normalize(wav)
    
    # Trim silence
    wav, _ = librosa.effects.trim(wav, top_db=30)
    
    sf.write(output_path, wav, target_sr)
    return wav

# --- Inference Helper ---

def run_inference(source, target, output_path, checkpoint, config, device_str, f0_condition=False, cfg_rate=0.7):
    import subprocess
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Ensure temp_out is clean before starting
    if os.path.exists("temp_out"):
        shutil.rmtree("temp_out")
    os.makedirs("temp_out", exist_ok=True)
        
    # Build command
    cmd = [
        sys.executable, "inference.py",
        "--source", str(source),
        "--target", str(target),
        "--output", "temp_out",
        "--config", str(config),
        "--diffusion-steps", "30",
        "--inference-cfg-rate", str(cfg_rate),
        "--fp16", "False"
    ]
    
    if f0_condition:
        cmd += ["--f0-condition", "True", "--auto-f0-adjust", "True"]
    else:
        cmd += ["--f0-condition", "False"]
    
    if checkpoint:
        cmd += ["--checkpoint", str(checkpoint)]
        
    print(f"Running: {' '.join(cmd)}")
    # Add timeout (5 minutes max for inference)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        print(f"Inference timed out after 5 minutes")
        return False
    
    if result.returncode != 0:
        print(f"Inference failed with error:\n{result.stderr}")
        return False
    
    # Find result - more robustly
    # Since we clean temp_out before starting, the only .wav file there is our result
    found = False
    if os.path.exists("temp_out"):
        wav_files = [f for f in os.listdir("temp_out") if f.endswith(".wav")]
        if wav_files:
            # Use the first wav file found
            target_file = os.path.join("temp_out", wav_files[0])
            if os.path.exists(output_path):
                os.remove(output_path)
            shutil.move(target_file, output_path)
            found = True
        else:
            print(f"  -> Error: No wav file found in temp_out. Inference might have failed.")
            if result.stdout: print(f"STDOUT: {result.stdout}")
            if result.stderr: print(f"STDERR: {result.stderr}")
        shutil.rmtree("temp_out")
        
    return found

# --- Config Helper ---

def get_config_for_checkpoint(ckpt_path, default_config):
    """
    Attempts to find the config file associated with a checkpoint.
    """
    if not ckpt_path or not os.path.exists(ckpt_path):
        return default_config
        
    ckpt_dir = os.path.dirname(ckpt_path)
    # Look for any .yml file in the directory
    for f in os.listdir(ckpt_dir):
        if f.endswith(".yml"):
            config_path = os.path.join(ckpt_dir, f)
            print(f"  -> Found config for {os.path.basename(ckpt_path)}: {f}")
            return config_path
            
    print(f"  -> Warning: No config found for {os.path.basename(ckpt_path)}, using default.")
    return default_config

# --- Main Comparison ---

def compare_all():
    print("--- High-Fidelity Model Evaluation (Target PESQ >= 3.2) ---")
    
    # Paths
    # Prefer WAV over MP3 for better PESQ
    source_audio_path = os.path.normpath("audio_inputs/user/test02.wav")
    if not os.path.exists(source_audio_path):
        inputs = [f for f in os.listdir("audio_inputs/user") if f.endswith(".wav") or f.endswith(".mp3")]
        if inputs: source_audio_path = os.path.normpath(os.path.join("audio_inputs/user", inputs[0]))

    target_audio_path = os.path.normpath("audio_inputs/reference/ref01_processed.wav") 
    
    output_dir = os.path.normpath("audio_outputs/comparison_report")
    os.makedirs(output_dir, exist_ok=True)
    
    # Checkpoints
    ckpt_russian = os.path.normpath("runs/russian_finetune_small_v3/ft_model.pth")
    ckpt_rapper = os.path.normpath("runs/rapper_oxxxy_finetune/ft_model.pth")
    
    # Configs
    config_small = os.path.normpath("configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml")
    config_base_44k = os.path.normpath("configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_campplus(device)
    
    # 1. High-Quality Preprocessing
    print(f"Preprocessing source: {source_audio_path}")
    source_cleaned_path = os.path.join(output_dir, "source_cleaned.wav")
    # Clean at 44.1kHz for the HQ model
    source_audio_44k = preprocess_audio_for_vc(source_audio_path, source_cleaned_path, target_sr=44100)
    # Reference at 16kHz for PESQ
    source_audio_16k, _ = librosa.load(source_cleaned_path, sr=16000)
    
    results = []
    
    # --- Test Cases ---
    test_cases = [
        # name, checkpoint, config, f0_condition, target_voice, cfg_rate
        ("Base Model (44kHz)", None, config_base_44k, True, target_audio_path, 1.0),
        ("Small Model (22kHz)", None, config_small, False, target_audio_path, 0.7),
        ("Russian FT (Small)", ckpt_russian, config_small, False, target_audio_path, 0.7),
        ("Rapper FT (Small)", ckpt_rapper, config_small, False, target_audio_path, 0.7),
        ("Rapper FT (Oxxxymiron)", ckpt_russian, config_small, False, os.path.normpath("datasets/rapper_finetune_preprocessed/oxxxymiron_dataset_raw_0070.wav"), 0.7),
    ]
    
    for name, ckpt, cfg, f0, target_path, cfg_rate in test_cases:
        print(f"\nTesting {name} (CFG: {cfg_rate})...")
        if ckpt and not os.path.exists(ckpt):
            results.append((name, 0.0, 0.0, 0.0, "Not Trained"))
            continue
            
        out_path = os.path.join(output_dir, f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.wav")
        
        # Use cleaned source for inference
        if run_inference(source_cleaned_path, target_path, out_path, ckpt, cfg, str(device), f0_condition=f0, cfg_rate=cfg_rate):
            # Load result at 16kHz for metrics
            out_wav, _ = librosa.load(out_path, sr=16000)
            
            # Metrics (Alignment is now automatic in project/preprocessing/evaluation.py)
            emb = get_embedding(out_path, encoder, device)
            
            # Calculate speaker similarity against the target voice used for this case
            current_target_emb = get_embedding(target_path, encoder, device)
            sim = get_cosine_sim(current_target_emb, emb)
            
            pesq_val = compute_pesq(source_audio_16k, out_wav, sr=16000)
            stoi_val = compute_stoi(source_audio_16k, out_wav, sr=16000)
            
            # Generate Mel Comparison Plot
            plot_path = os.path.join(output_dir, f"mel_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png")
            plot_mel_comparison(source_cleaned_path, target_path, out_path, plot_path, title_suffix=f"({name})")
            
            results.append({
                "name": name,
                "pesq": float(pesq_val),
                "stoi": float(stoi_val),
                "speaker_similarity": float(sim),
                "status": "OK",
                "output_path": out_path,
                "spectrogram_path": plot_path
            })
        else:
            results.append({
                "name": name,
                "pesq": 0.0,
                "stoi": 0.0,
                "speaker_similarity": 0.0,
                "status": "Failed",
                "output_path": None,
                "spectrogram_path": None
            })

    # --- Final Report ---
    print("\n" + "="*110)
    print(f"{'Method/Model':<25} | {'PESQ':<10} | {'STOI':<10} | {'Speaker Similarity':<20} | {'Status'}")
    print("-" * 110)
    
    for r in results:
        print(f"{r['name']:<25} | {r['pesq']:.4f}{' '*4} | {r['stoi']:.4f}{' '*4} | {r['speaker_similarity']:.4f}{' '*12} | {r['status']}")
        
    print("-" * 110)
    
    # Save results to JSON
    json_path = os.path.join(output_dir, "comparison_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nDetailed JSON report saved to: {json_path}")
    print("PESQ Analysis Note:")
    print("1. 'Reconstruction' shows the model's quality limit when speaker identity is preserved.")
    print("2. 'Conversion' PESQ is naturally lower because PESQ penalizes speaker identity changes (timbre shift).")
    print("3. For PESQ > 3.2, look at 'Reconstruction (Base)' result.")
    print(f"\nResults saved in: {output_dir}")

if __name__ == "__main__":
    compare_all()
