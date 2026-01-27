"""
Quality metrics for voice conversion evaluation:
- Speaker similarity (SECS)
- Pitch preservation (F0 correlation, F0 RMSE)
- STOI score
"""

import numpy as np
import torch
try:
    import torchaudio
except (ImportError, OSError):
    torchaudio = None  # Optional dependency
import librosa
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    HAS_RESEMBLYZER = True
except ImportError:
    HAS_RESEMBLYZER = False
    print("[Warning] resemblyzer not available. Speaker similarity will be limited.")

try:
    from pystoi import stoi
    HAS_STOI = True
except ImportError:
    HAS_STOI = False
    print("[Warning] pystoi not available. STOI will not be computed.")

try:
    from pesq import pesq
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False


class QualityMetrics:
    """Compute quality metrics for voice conversion"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.encoder = None
        if HAS_RESEMBLYZER:
            try:
                self.encoder = VoiceEncoder(device=device if device != 'cuda' else 'cpu')
            except:
                self.encoder = None
    
    def compute_speaker_similarity(self, reference_audio: np.ndarray, 
                                   converted_audio: np.ndarray,
                                   sample_rate: int = 16000) -> float:
        """
        Compute Speaker Embedding Cosine Similarity (SECS)
        
        Args:
            reference_audio: Reference speaker audio (numpy array)
            converted_audio: Converted audio (numpy array)
            sample_rate: Sample rate of both audios
        
        Returns:
            Similarity score between 0 and 1 (higher is better)
        """
        if not HAS_RESEMBLYZER or self.encoder is None:
            # Fallback: simple spectral similarity
            ref_spec = np.abs(librosa.stft(reference_audio, n_fft=2048))
            conv_spec = np.abs(librosa.stft(converted_audio, n_fft=2048))
            
            # Normalize
            ref_spec = ref_spec / (np.linalg.norm(ref_spec) + 1e-8)
            conv_spec = conv_spec / (np.linalg.norm(conv_spec) + 1e-8)
            
            # Cosine similarity
            similarity = np.dot(ref_spec.flatten(), conv_spec.flatten())
            return float(np.clip(similarity, 0, 1))
        
        try:
            # Resample to 16kHz if needed (resemblyzer requirement)
            if sample_rate != 16000:
                ref_16k = librosa.resample(reference_audio, orig_sr=sample_rate, target_sr=16000)
                conv_16k = librosa.resample(converted_audio, orig_sr=sample_rate, target_sr=16000)
            else:
                ref_16k = reference_audio
                conv_16k = converted_audio
            
            # Preprocess for resemblyzer
            ref_wav = preprocess_wav(ref_16k)
            conv_wav = preprocess_wav(conv_16k)
            
            # Get embeddings
            ref_embed = self.encoder.embed_utterance(ref_wav)
            conv_embed = self.encoder.embed_utterance(conv_wav)
            
            # Cosine similarity
            similarity = np.inner(ref_embed, conv_embed)
            return float(np.clip(similarity, -1, 1))
        except Exception as e:
            print(f"[Warning] Speaker similarity computation failed: {e}")
            return 0.0
    
    def extract_f0(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract F0 (fundamental frequency) contour using librosa
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate
        
        Returns:
            F0 values in Hz
        """
        try:
            # Use pyin for robust F0 estimation
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
                sr=sample_rate
            )
            # Replace NaN with 0
            f0 = np.nan_to_num(f0, nan=0.0)
            return f0
        except Exception as e:
            # Fallback to simpler method
            try:
                f0 = librosa.yin(audio, fmin=80, fmax=400, sr=sample_rate)
                return np.nan_to_num(f0, nan=0.0)
            except:
                return np.zeros(len(audio) // 512)  # Rough estimate
    
    def compute_pitch_metrics(self, reference_audio: np.ndarray,
                            converted_audio: np.ndarray,
                            source_audio: np.ndarray,
                            sample_rate: int = 16000) -> Tuple[float, float]:
        """
        Compute pitch preservation metrics
        
        Args:
            reference_audio: Target speaker reference audio
            converted_audio: Converted audio
            source_audio: Original source audio
        
        Returns:
            (f0_correlation, f0_rmse)
            - f0_correlation: Correlation between source and converted F0 (higher is better, -1 to 1)
            - f0_rmse: RMSE of F0 difference (lower is better, in Hz)
        """
        try:
            # Extract F0 contours
            source_f0 = self.extract_f0(source_audio, sample_rate)
            converted_f0 = self.extract_f0(converted_audio, sample_rate)
            
            # Align lengths
            min_len = min(len(source_f0), len(converted_f0))
            source_f0 = source_f0[:min_len]
            converted_f0 = converted_f0[:min_len]
            
            # Remove unvoiced frames (F0 = 0)
            voiced_mask = (source_f0 > 0) & (converted_f0 > 0)
            if voiced_mask.sum() < 10:  # Need at least some voiced frames
                return 0.0, 1000.0
            
            source_f0_voiced = source_f0[voiced_mask]
            converted_f0_voiced = converted_f0[voiced_mask]
            
            # Correlation
            if len(source_f0_voiced) > 1:
                correlation = np.corrcoef(source_f0_voiced, converted_f0_voiced)[0, 1]
                correlation = np.nan_to_num(correlation, nan=0.0)
            else:
                correlation = 0.0
            
            # RMSE
            rmse = np.sqrt(np.mean((source_f0_voiced - converted_f0_voiced) ** 2))
            rmse = np.nan_to_num(rmse, nan=1000.0)
            
            return float(correlation), float(rmse)
        except Exception as e:
            print(f"[Warning] Pitch metrics computation failed: {e}")
            return 0.0, 1000.0
    
    def compute_stoi(self, reference_audio: np.ndarray,
                    converted_audio: np.ndarray,
                    sample_rate: int = 16000) -> float:
        """
        Compute STOI (Short-Time Objective Intelligibility) score
        
        Args:
            reference_audio: Reference audio (usually source)
            converted_audio: Converted audio
            sample_rate: Sample rate
        
        Returns:
            STOI score between 0 and 1 (higher is better)
        """
        if not HAS_STOI:
            return 0.0
        
        try:
            # Ensure same length
            min_len = min(len(reference_audio), len(converted_audio))
            ref = reference_audio[:min_len]
            conv = converted_audio[:min_len]
            
            # STOI requires specific sample rates
            if sample_rate not in [8000, 10000, 16000]:
                # Resample to 16kHz
                ref = librosa.resample(ref, orig_sr=sample_rate, target_sr=16000)
                conv = librosa.resample(conv, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            
            score = stoi(ref, conv, sample_rate, extended=False)
            return float(np.clip(score, 0, 1))
        except Exception as e:
            print(f"[Warning] STOI computation failed: {e}")
            return 0.0
    
    def compute_pesq(self, reference_audio: np.ndarray,
                    converted_audio: np.ndarray,
                    sample_rate: int = 16000) -> float:
        """
        Compute PESQ score (optional, for completeness)
        
        Args:
            reference_audio: Reference audio
            converted_audio: Converted audio
            sample_rate: Sample rate
        
        Returns:
            PESQ score (higher is better, typically -0.5 to 4.5)
        """
        if not HAS_PESQ:
            return 0.0
        
        try:
            # Ensure same length
            min_len = min(len(reference_audio), len(converted_audio))
            ref = reference_audio[:min_len]
            conv = converted_audio[:min_len]
            
            # PESQ requires 8kHz or 16kHz
            if sample_rate not in [8000, 16000]:
                ref = librosa.resample(ref, orig_sr=sample_rate, target_sr=16000)
                conv = librosa.resample(conv, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            
            mode = 'wb' if sample_rate == 16000 else 'nb'
            score = pesq(sample_rate, ref, conv, mode)
            return float(score)
        except Exception as e:
            print(f"[Warning] PESQ computation failed: {e}")
            return 0.0
