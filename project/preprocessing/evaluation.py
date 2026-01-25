import numpy as np
import warnings
import librosa
import scipy.signal

def align_audio(ref: np.ndarray, deg: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Robustly aligns two audio signals using cross-correlation on onset envelopes.
    This is much more reliable than raw signal correlation for voice conversion.
    """
    # 1. Normalize and Trim
    ref = librosa.util.normalize(ref)
    deg = librosa.util.normalize(deg)
    ref, _ = librosa.effects.trim(ref, top_db=25)
    deg, _ = librosa.effects.trim(deg, top_db=25)
    
    # 2. Compute Onset Envelopes (robust to timbre/speaker changes)
    # We use a lower sr for faster correlation
    hop_len = 256
    ref_env = librosa.onset.onset_strength(y=ref, sr=sr, hop_length=hop_len)
    deg_env = librosa.onset.onset_strength(y=deg, sr=sr, hop_length=hop_len)
    
    # 3. Find Optimal Lag via Cross-Correlation
    correlation = scipy.signal.correlate(ref_env, deg_env, mode='full')
    lags = scipy.signal.correlation_lags(len(ref_env), len(deg_env), mode='full')
    
    best_lag_frames = lags[np.argmax(correlation)]
    best_lag_samples = int(best_lag_frames * hop_len)
    
    # 4. Apply Lag
    if best_lag_samples > 0:
        # deg is 'late' compared to ref, shift it left (cut start or pad ref)
        aligned_deg = np.pad(deg, (best_lag_samples, 0))
    elif best_lag_samples < 0:
        # deg is 'early' compared to ref, shift it right
        aligned_deg = deg[-best_lag_samples:]
    else:
        aligned_deg = deg
        
    # 5. Match Lengths exactly (PESQ requirement)
    min_len = min(len(ref), len(aligned_deg))
    ref = ref[:min_len]
    aligned_deg = aligned_deg[:min_len]
    
    return ref, aligned_deg

def compute_pesq(ref: np.ndarray, deg: np.ndarray, sr: int = 16000) -> float:
    """
    Compute PESQ score with automatic alignment. Requires 16kHz or 8kHz sample rate.
    """
    try:
        from pesq import pesq
        
        # Automatic alignment (PESQ is extremely sensitive to time shifts)
        ref, deg = align_audio(ref, deg, sr)
        
        if sr not in [8000, 16000]:
            # This is handled outside usually, but for safety:
            ref = librosa.resample(ref, orig_sr=sr, target_sr=16000)
            deg = librosa.resample(deg, orig_sr=sr, target_sr=16000)
            sr = 16000
            
        # Mode 'wb' (wideband) for 16k, 'nb' (narrowband) for 8k
        mode = 'wb' if sr == 16000 else 'nb'
        return pesq(sr, ref, deg, mode)
    except ImportError:
        return 0.0
    except Exception as e:
        print(f"PESQ computation failed: {e}")
        return 0.0

def compute_stoi(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """
    Compute STOI score with automatic alignment.
    """
    try:
        from pystoi import stoi
        # Align
        ref, deg = align_audio(ref, deg, sr)
        return stoi(ref, deg, sr, extended=False)
    except ImportError:
        return 0.0
    except Exception as e:
        print(f"STOI computation failed: {e}")
        return 0.0
