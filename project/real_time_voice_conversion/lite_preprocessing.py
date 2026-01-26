import numpy as np
import scipy.signal

class LitePreprocessor:
    """
    Lightweight audio preprocessor optimized for real-time edge devices (Jetson Nano).
    Focuses on low latency (<5ms) and CPU efficiency.
    """
    def __init__(self, sample_rate=22050, noise_gate_db=-40.0, high_pass_freq=80.0):
        self.sample_rate = sample_rate
        self.noise_gate_threshold = 10 ** (noise_gate_db / 20)
        
        # Design High-pass filter (Butterworth 2nd order)
        # Removes low-frequency rumble/wind noise
        nyquist = 0.5 * sample_rate
        norm_cutoff = high_pass_freq / nyquist
        self.b, self.a = scipy.signal.butter(2, norm_cutoff, btype='high', analog=False)
        
        # State for streaming filter (zi)
        self.zi = scipy.signal.lfilter_zi(self.b, self.a)
        
    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Process a chunk of audio (samples,).
        1. High-pass Filter
        2. Noise Gate
        """
        # Ensure 1D array
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.flatten()
            
        # 1. High-pass Filter (Streaming)
        # Use lfilter with state to maintain continuity between chunks
        filtered_chunk, self.zi = scipy.signal.lfilter(self.b, self.a, audio_chunk, zi=self.zi)
        
        # 2. Noise Gate (Simple Hard Gate with no attack/release for zero latency)
        # Calculate RMS of the chunk
        rms = np.sqrt(np.mean(filtered_chunk**2))
        
        if rms < self.noise_gate_threshold:
            # If below threshold, mute completely
            # For smoother gating, one could implement a soft knee, but hard gate is fastest
            return np.zeros_like(filtered_chunk)
            
        return filtered_chunk
