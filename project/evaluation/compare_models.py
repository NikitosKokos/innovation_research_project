import os
import sys
import torch
import librosa
import torchaudio
import torchaudio.transforms as T
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from modules.campplus.DTDNN import CAMPPlus
from hf_utils import load_custom_model_from_hf

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
        wav = T.Resample(sr, 16000)(wav)
        
    if wav.ndim == 1: wav = wav.unsqueeze(0)
    
    wav = wav.to(device)
    feat = torchaudio.compliance.kaldi.fbank(wav, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    
    with torch.no_grad():
        emb = model(feat.unsqueeze(0))
    return emb

def get_cosine_sim(emb1, emb2):
    return torch.nn.CosineSimilarity(dim=1)(emb1, emb2).item()

# --- Inference Helper ---

def run_inference(source, target, output_path, checkpoint, config, device_str):
    if os.path.exists(output_path):
        os.remove(output_path)
        
    cmd = f"python inference.py --source \"{source}\" --target \"{target}\" --output \"temp_out\" --config \"{config}\" --diffusion-steps 30 --inference-cfg-rate 0.7 --fp16 False"
    
    if checkpoint:
        cmd += f" --checkpoint \"{checkpoint}\""
        
    print(f"Running inference...")
    os.system(cmd)
    
    # Find result
    src_base = os.path.basename(source).split(".")[0]
    tgt_base = os.path.basename(target).split(".")[0]
    pattern = f"vc_{src_base}_{tgt_base}"
    
    if not os.path.exists("temp_out"): return False
    
    found = False
    for f in os.listdir("temp_out"):
        if pattern in f and f.endswith(".wav"):
            os.rename(os.path.join("temp_out", f), output_path)
            found = True
            break
            
    if os.path.exists("temp_out"):
        import shutil
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
    print("--- Model Comparison & Metrics ---")
    
    # Paths
    source_audio = "audio_inputs/user/test02.wav"
    # Try to find a source if default doesn't exist
    if not os.path.exists(source_audio):
         inputs = [f for f in os.listdir("audio_inputs/user") if f.endswith(".wav")]
         if inputs: source_audio = os.path.join("audio_inputs/user", inputs[0])

    target_audio = "audio_inputs/reference/ref01_processed.wav" # The target voice
    
    output_dir = "audio_outputs/comparison_report"
    os.makedirs(output_dir, exist_ok=True)
    
    # Checkpoints
    ckpt_russian = "runs/run_dit_mel_seed_uvit_whisper_small_wavenet/russian_finetune_small_v3/ft_model.pth"
    ckpt_rapper = "runs/rapper_finetune/ft_model.pth"
    
    # Default Config (Small/Whisper) - used for Original and fallback
    default_config = "configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_campplus(device)
    
    target_emb = get_embedding(target_audio, encoder, device)
    
    results = []
    
    # 1. Original Model
    print("\n[1/3] Testing Original Model...")
    out_orig = os.path.join(output_dir, "original.wav")
    if run_inference(source_audio, target_audio, out_orig, None, default_config, str(device)):
        emb = get_embedding(out_orig, encoder, device)
        sim = get_cosine_sim(target_emb, emb)
        results.append(("Original", sim, out_orig))
    else:
        results.append(("Original", 0.0, "Failed"))

    # 2. Russian Fine-tuned
    print("\n[2/3] Testing Russian Fine-tuned...")
    if os.path.exists(ckpt_russian):
        out_rus = os.path.join(output_dir, "russian_ft.wav")
        # Auto-detect config
        rus_config = get_config_for_checkpoint(ckpt_russian, default_config)
        
        if run_inference(source_audio, target_audio, out_rus, ckpt_russian, rus_config, str(device)):
            emb = get_embedding(out_rus, encoder, device)
            sim = get_cosine_sim(target_emb, emb)
            results.append(("Russian FT", sim, out_rus))
        else:
            results.append(("Russian FT", 0.0, "Failed"))
    else:
        results.append(("Russian FT", 0.0, "Not Trained"))

    # 3. Rapper Fine-tuned
    print("\n[3/3] Testing Rapper Fine-tuned...")
    if os.path.exists(ckpt_rapper):
        out_rap = os.path.join(output_dir, "rapper_ft.wav")
        # Auto-detect config
        rap_config = get_config_for_checkpoint(ckpt_rapper, default_config)
        
        if run_inference(source_audio, target_audio, out_rap, ckpt_rapper, rap_config, str(device)):
            emb = get_embedding(out_rap, encoder, device)
            sim = get_cosine_sim(target_emb, emb)
            results.append(("Rapper FT", sim, out_rap))
        else:
            results.append(("Rapper FT", 0.0, "Failed"))
    else:
        results.append(("Rapper FT", 0.0, "Not Trained"))
        
    # Report
    print("\n" + "="*60)
    print("COMPARISON REPORT")
    print("="*60)
    print(f"{'Model':<20} | {'Cosine Sim (Higher is Better)':<30} | {'Status'}")
    print("-" * 60)
    
    for name, sim, path in results:
        status = "OK" if path and path != "Failed" and path != "Not Trained" else path
        print(f"{name:<20} | {sim:.4f}{' '*24} | {status}")
        
    print("-" * 60)
    print(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    compare_all()
