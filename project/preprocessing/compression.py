import numpy as np

class DynamicRangeCompressor:
    def __init__(self, threshold_db=-20.0, ratio=4.0, attack_ms=5.0, release_ms=50.0, sr=22050):
        self.threshold_db = threshold_db
        self.ratio = ratio
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.sr = sr
        
        # Calculate coefficients
        self.attack_coef = np.exp(-1000 / (attack_ms * sr))
        self.release_coef = np.exp(-1000 / (release_ms * sr))

    def compress(self, audio: np.ndarray) -> np.ndarray:
        # Convert to dB
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        audio_abs = np.abs(audio)
        audio_db = 20 * np.log10(audio_abs + eps)
        
        # Calculate gain reduction
        # This is a hard-knee compressor logic for gain calculation
        over_threshold = audio_db - self.threshold_db
        gain_reduction_db = np.zeros_like(audio_db)
        
        mask = over_threshold > 0
        gain_reduction_db[mask] = over_threshold[mask] * (1 - 1/self.ratio)
        
        # Apply attack/release envelope to gain reduction
        # We want to smooth the gain reduction curve
        # Note: Usually compression is done on a side-chain signal or envelope of the signal
        
        # 1. Compute envelope of the signal
        # Simple peak envelope follower
        envelope = np.zeros_like(audio_abs)
        current_env = 0.0
        
        # Vectorized implementation is hard for recursive filter, using loop for correctness
        # Optimizing: using python loop is slow. 
        # For efficiency in python, we can use scipy lfilter if possible, but attack/release are different.
        # Let's use a simplified block-based or approximated approach for speed if needed, 
        # but for reference processing (offline), loop is acceptable if file is not huge.
        
        # However, for pure python, let's try to be efficient.
        # Let's compute gain reduction target first based on static curve
        
        target_gain_db = -gain_reduction_db 
        
        # Smoothing the gain (attack/release)
        # If target_gain < current_gain, we are attacking (gain is dropping)
        # If target_gain > current_gain, we are releasing (gain is recovering to 0dB)
        
        smoothed_gain_db = np.zeros_like(target_gain_db)
        current_g = 0.0
        
        # We process in chunks to avoid huge loops if possible, or just use numba if available.
        # Given constraints, we'll implement a basic loop. It handles 5-10s clips fine.
        
        for i in range(len(target_gain_db)):
            target = target_gain_db[i]
            if target < current_g:
                # Attack phase (gain reducing)
                current_g = self.attack_coef * current_g + (1 - self.attack_coef) * target
            else:
                # Release phase (gain increasing)
                current_g = self.release_coef * current_g + (1 - self.release_coef) * target
            smoothed_gain_db[i] = current_g
            
        # Apply gain
        gain_linear = 10 ** (smoothed_gain_db / 20)
        compressed_audio = audio * gain_linear
        
        # Make up gain? Usually compressor has makeup gain.
        # The guide doesn't explicitly ask for makeup gain, but normalization usually happens before or after.
        # If normalization is before, compression reduces volume. 
        # We will leave it without auto makeup gain as normalization might happen again or user sets it.
        # But typically we want to peak normalize after compression.
        
        return compressed_audio
