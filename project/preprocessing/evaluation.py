import numpy as np
import warnings
import librosa

def compute_pesq(ref: np.ndarray, deg: np.ndarray, sr: int = 16000) -> float:
    """
    Compute PESQ score. Requires 16kHz or 8kHz sample rate.
    """
    try:
        from pesq import pesq
        if sr not in [8000, 16000]:
            # Resample if needed
            if sr != 16000:
                 # This might be slow inside a loop, better to handle outside
                 pass
            
        # PESQ requires 16k or 8k. 
        # Mode 'wb' (wideband) for 16k, 'nb' (narrowband) for 8k
        mode = 'wb' if sr == 16000 else 'nb'
        return pesq(sr, ref, deg, mode)
    except ImportError:
        # warnings.warn("pesq package not found. Returning 0.0")
        return 0.0
    except Exception as e:
        print(f"PESQ computation failed: {e}")
        return 0.0

def compute_stoi(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """
    Compute STOI score.
    """
    try:
        from pystoi import stoi
        return stoi(ref, deg, sr, extended=False)
    except ImportError:
        # warnings.warn("pystoi package not found. Returning 0.0")
        return 0.0
    except Exception as e:
        print(f"STOI computation failed: {e}")
        return 0.0
