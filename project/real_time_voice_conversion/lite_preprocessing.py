import numpy as np
import scipy.signal

class LitePreprocessor:
    """
    Lightweight audio preprocessor optimized for real-time edge devices (Jetson Nano).
    Focuses on low latency (<5ms) and CPU efficiency.
    """
    def __init__(self, sample_rate=22050, noise_gate_db=-55.0, high_pass_freq=80.0):
        self.sample_rate = sample_rate
        self.noise_gate_threshold = 10 ** (noise_gate_db / 20)
        
        # Design High-pass filter (Butterworth 2nd order)
        # Removes low-frequency rumble/wind noise
        nyquist = 0.5 * sample_rate
        norm_cutoff = high_pass_freq / nyquist
        self.b, self.a = scipy.signal.butter(2, norm_cutoff, btype='high', analog=False)
        
        # State for streaming filter (zi)
        self.zi = scipy.signal.lfilter_zi(self.b, self.a)
        
        # Noise gate state for soft attack (prevents cutting off speech starts)
        self.gate_state = 0.0  # 0 = closed, 1 = open
        self.attack_rate = 0.3  # How fast to open (higher = faster, but more sensitive to noise)
        self.release_rate = 0.1  # How fast to close (lower = slower release)
        
    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Process a chunk of audio (samples,).
        1. High-pass Filter
        2. Noise Gate (with soft attack to prevent cutting off speech starts)
        """
        # Ensure 1D array
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.flatten()
            
        # 1. High-pass Filter (Streaming)
        # Use lfilter with state to maintain continuity between chunks
        filtered_chunk, self.zi = scipy.signal.lfilter(self.b, self.a, audio_chunk, zi=self.zi)
        
        # 2. Noise Gate (Soft Gate with attack/release to catch speech starts)
        # Calculate RMS of the chunk
        rms = np.sqrt(np.mean(filtered_chunk**2))
        
        # Determine if we should open or close the gate
        if rms >= self.noise_gate_threshold:
            # Audio detected - open gate (with attack)
            self.gate_state = min(1.0, self.gate_state + self.attack_rate)
        else:
            # Silence - close gate (with release)
            self.gate_state = max(0.0, self.gate_state - self.release_rate)
        
        # Apply gate with smooth transition
        # This prevents cutting off the start of speech
        gated_chunk = filtered_chunk * self.gate_state
        
        return gated_chunk
