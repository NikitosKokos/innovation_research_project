import numpy as np
import soundfile as sf
import warnings

class LUFSNormalizer:
    def __init__(self, target_lufs=-23.0, sr=22050):
        self.target_lufs = target_lufs
        self.sr = sr
        self.has_pyloudnorm = False
        
        try:
            import pyloudnorm as pyln
            self.meter = pyln.Meter(sr)
            self.pyln = pyln
            self.has_pyloudnorm = True
        except ImportError:
            warnings.warn("pyloudnorm not found. Using RMS normalization as fallback.")
            
    def normalize(self, audio: np.ndarray) -> np.ndarray:
        if self.has_pyloudnorm:
            try:
                # pyloudnorm expects shape (samples, channels)
                # if mono, it works with (samples,) or (samples, 1)
                
                # Measure integrated loudness
                loudness = self.meter.integrated_loudness(audio)
                
                # Normalize
                normalized_audio = self.pyln.normalize.loudness(audio, loudness, self.target_lufs)
                return normalized_audio
            except Exception as e:
                print(f"LUFS normalization failed: {e}. Fallback to RMS.")
                return self._rms_normalize(audio)
        else:
            return self._rms_normalize(audio)
            
    def _rms_normalize(self, audio: np.ndarray) -> np.ndarray:
        # Simple RMS normalization to approximate -23 LUFS
        # -23 LUFS is roughly -23 dBFS RMS for simple signals, but it varies.
        # We will target -23 dB RMS relative to full scale.
        
        rms = np.sqrt(np.mean(audio**2))
        if rms == 0:
            return audio
            
        # Target linear amplitude
        target_linear = 10 ** (self.target_lufs / 20)
        
        gain = target_linear / (rms + 1e-10)
        return audio * gain

