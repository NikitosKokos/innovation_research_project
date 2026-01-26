import sounddevice as sd
import numpy as np
import queue
import threading
import time
import sys
from .lite_preprocessing import LitePreprocessor

class VoiceConversionStream:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        
        # Lite Preprocessor for Edge Optimization
        self.preprocessor = LitePreprocessor(
            sample_rate=config.SAMPLE_RATE,
            noise_gate_db=-45.0, # Adjustable threshold
            high_pass_freq=80.0
        )
        
        # Queues for passing audio between threads
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        
        self.running = False
        self.stream = None
        self.processing_thread = None
        
        # Statistics
        self.total_latency = []
        
    def _audio_callback(self, indata, outdata, frames, time_info, status):
        """
        SoundDevice callback.
        1. Reads input audio (indata) -> Puts into input_queue
        2. Reads processed audio from output_queue -> Writes to output (outdata)
        """
        if status:
            print(f"[Stream Status] {status}", file=sys.stderr)
            
        # 1. Handle Input
        # Copy input data to queue (must make a copy!)
        self.input_queue.put(indata.copy())
        
        # 2. Handle Output
        try:
            # Try to get processed audio from output queue
            # non-blocking get; if empty, play silence
            data = self.output_queue.get_nowait()
            
            # Ensure data length matches buffer size
            if len(data) < len(outdata):
                # Pad with zeros if too short
                outdata[:len(data)] = data
                outdata[len(data):] = 0
            elif len(data) > len(outdata):
                # Truncate if too long
                outdata[:] = data[:len(outdata)]
            else:
                outdata[:] = data
                
        except queue.Empty:
            # Underflow: Model hasn't finished processing yet
            # Play silence
            outdata[:] = 0
            
    def _processing_loop(self):
        """
        Runs in a separate thread.
        Takes audio from input_queue -> Runs Model -> Puts into output_queue
        """
        print("[Stream] Processing thread started.")
        while self.running:
            try:
                # Get input chunk (blocking with timeout to check self.running)
                try:
                    input_chunk = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Measure processing time
                start_time = time.time()
                
                # --- INFERENCE ---
                # Flatten input chunk (channels, samples) -> (samples,)
                input_mono = input_chunk.flatten()
                
                # 1. Lite Preprocessing (Noise Gate + High Pass)
                input_mono = self.preprocessor.process(input_mono)
                
                # Run Voice Conversion
                converted_chunk = self.model.process_chunk(input_mono)
                
                # Reshape for output (samples, channels)
                # Assuming mono output from model, but sounddevice expects (samples, channels)
                output_chunk = converted_chunk.reshape(-1, 1)
                
                # -----------------
                
                process_time = (time.time() - start_time) * 1000
                # print(f"Inference Time: {process_time:.1f}ms") # Debug logging
                
                self.output_queue.put(output_chunk)
                
            except Exception as e:
                print(f"[Stream Error] {e}")
                
    def start(self):
        if self.running:
            return

        print(f"[Stream] Starting audio stream (Blocksize: {self.config.BLOCK_SIZE})...")
        self.running = True
        
        # Start Processing Thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        
        # Start Audio Stream
        # Find device IDs from config names
        try:
            # Simple logic: if config says 'pulse', let sounddevice find default
            # If it's an integer ID, use it.
            in_dev = self.config.INPUT_DEVICE
            out_dev = self.config.OUTPUT_DEVICE
            
            # If names are 'pulse' or 'default', sounddevice usually handles them if we pass None or specific index
            # But let's try passing them directly if they are strings, or IDs if ints.
            
            self.stream = sd.Stream(
                device=(in_dev, out_dev),
                samplerate=self.config.SAMPLE_RATE,
                blocksize=self.config.BLOCK_SIZE,
                channels=1, # Mono
                callback=self._audio_callback
            )
            self.stream.start()
            print("[Stream] Audio stream active.")
            
        except Exception as e:
            print(f"[Stream Error] Failed to start stream: {e}")
            self.running = False
            
    def stop(self):
        if not self.running:
            return
            
        print("[Stream] Stopping...")
        self.running = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            
        if self.processing_thread:
            self.processing_thread.join()
            
        print("[Stream] Stopped.")
