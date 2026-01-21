import numpy as np
import scipy.signal
import librosa
import warnings

class BaseDenoiser:
    def denoise(self, audio: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class NoisereduceDenoiser(BaseDenoiser):
    """
    Wrapper around the 'noisereduce' library which implements spectral gating 
    (spectral subtraction with time-frequency smoothing).
    This is generally more robust than simple spectral subtraction.
    """
    def __init__(self, sr=22050, prop_decrease=1.0, stationary=True):
        self.sr = sr
        self.prop_decrease = prop_decrease
        self.stationary = stationary
        try:
            import noisereduce as nr
            self.nr = nr
            self.available = True
        except ImportError:
            warnings.warn("noisereduce library not found. Falling back to simple Spectral Subtraction.")
            self.available = False
            self.fallback = SpectralSubtractionDenoiser(sr=sr)

    def denoise(self, audio: np.ndarray) -> np.ndarray:
        if self.available:
            # noisereduce expects shape (channels, samples) or (samples,)
            # standard usage: reduce_noise(y=audio_data, sr=sample_rate)
            try:
                # stationary=True uses a statistical approach to find the noise profile across the whole file
                return self.nr.reduce_noise(y=audio, sr=self.sr, stationary=self.stationary, prop_decrease=self.prop_decrease)
            except Exception as e:
                print(f"noisereduce failed: {e}. using fallback.")
                return self.fallback.denoise(audio)
        else:
            return self.fallback.denoise(audio)

class WienerDenoiser(BaseDenoiser):
    def __init__(self, sr=22050):
        self.sr = sr

    def denoise(self, audio: np.ndarray) -> np.ndarray:
        # Improved Wiener Filter Implementation
        # 1. STFT
        # 2. Estimate Noise Power Spectral Density (PSD) using Minimum Statistics
        # 3. Compute Wiener Gain
        # 4. ISTFT
        
        n_fft = 2048
        hop_length = 512
        win_length = 2048
        
        # Pad audio to avoid edge effects
        padded_audio = np.pad(audio, (0, n_fft), mode='constant')
        
        S = librosa.stft(padded_audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        S_abs = np.abs(S)
        S_phase = np.angle(S)
        
        # Power Spectrogram
        P_y = S_abs ** 2
        
        # Estimate Noise PSD using Minimum Statistics approximation
        # We assume noise is stationary and present in the "quietest" frames
        # Sliding window minimum or quantile based
        
        # Simple approach: Take 10th percentile of energy across time for each frequency bin
        # This assumes at least 10% of the signal is noise-only or low SNR
        P_n_est = np.percentile(P_y, 10, axis=1, keepdims=True)
        
        # A priori SNR estimation (Decision-Directed approach could be better but keeping it simple)
        # P_s_est = max(P_y - P_n_est, 0)
        P_s_est = np.maximum(P_y - P_n_est, 1e-10)
        
        # Wiener Gain: G(f, t) = P_s(f, t) / (P_s(f, t) + P_n(f))
        # Smoothing parameter for SNR to avoid musical noise
        
        G = P_s_est / (P_s_est + P_n_est + 1e-10)
        
        # Apply Gain
        S_clean = S_abs * G * np.exp(1j * S_phase)
        
        clean_audio = librosa.istft(S_clean, hop_length=hop_length, win_length=win_length)
        
        # Trim padding
        return clean_audio[:len(audio)]

class SpectralSubtractionDenoiser(BaseDenoiser):
    def __init__(self, sr=22050):
        self.sr = sr

    def denoise(self, audio: np.ndarray) -> np.ndarray:
        # Improved Spectral Subtraction
        n_fft = 2048
        hop_length = 512
        
        S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        S_mag = np.abs(S)
        S_phase = np.angle(S)
        
        # Noise Estimation: Average of lowest energy frames
        frame_energy = np.sum(S_mag ** 2, axis=0)
        
        # Identify noise frames (lowest 10%)
        num_frames = len(frame_energy)
        num_noise_frames = max(int(num_frames * 0.1), 5)
        
        sorted_indices = np.argsort(frame_energy)
        noise_indices = sorted_indices[:num_noise_frames]
        
        noise_mag = np.mean(S_mag[:, noise_indices], axis=1, keepdims=True)
        
        # Spectral Subtraction with spectral floor
        alpha = 2.0  # subtraction factor (aggressiveness)
        beta = 0.02  # spectral floor (prevent creating holes)
        
        S_clean_mag = S_mag - (alpha * noise_mag)
        S_clean_mag = np.maximum(S_clean_mag, beta * noise_mag) # Spectral floor
        
        S_clean = S_clean_mag * np.exp(1j * S_phase)
        return librosa.istft(S_clean, length=len(audio))

def get_denoiser(method: str, sr: int = 22050) -> BaseDenoiser:
    if method.lower() == 'noisereduce':
        return NoisereduceDenoiser(sr=sr)
    elif method.lower() == 'wiener':
        return WienerDenoiser(sr=sr)
    elif method.lower() == 'spectral_subtraction':
        return SpectralSubtractionDenoiser(sr=sr)
    else:
        # Default to noisereduce if unknown or use fallback
        return NoisereduceDenoiser(sr=sr)
