import sounddevice as sd
import numpy as np
import queue
import threading
import time
import sys
# Lite preprocessing disabled for now
# from lite_preprocessing import LitePreprocessor

class VoiceConversionStream:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        
        # Lite Preprocessing DISABLED for now
        # self.preprocessor = LitePreprocessor(
        #     sample_rate=config.SAMPLE_RATE,
        #     noise_gate_db=-55.0,
        #     high_pass_freq=80.0
        # )
        self.use_lite_preprocessing = False
        
        # Queues for passing audio between threads
        # Input queue: Larger to allow buffering full sentences while processing happens
        # Output queue: Larger to buffer processed audio
        # Maxsize set to 5000 (~2 minutes buffer) to prevent dropping audio during long processing
        self.input_queue = queue.Queue(maxsize=5000)
        self.output_queue = queue.Queue(maxsize=5000)
        self.last_output_chunk = None  # Store last output chunk to repeat if queue is empty
        
        # Chunk accumulation buffer (to provide sufficient context for model)
        self.chunk_buffer = []
        self.chunks_to_accumulate = getattr(config, 'CHUNKS_TO_ACCUMULATE', 129)
        self.last_input_time = time.time()
        self.silence_frames = 0
        
        # Test tone for debugging audio output
        self.test_tone_phase = 0.0
        self.test_tone_freq = 440.0  # A4 note
        
        self.running = False
        self.stream = None
        self.processing_thread = None
        
        # Process chunks individually for best quality (no accumulation to avoid artifacts)
        # Accumulation was causing speech to become a mess
        # self.chunk_buffer = []  # Buffer for minimal context (optional) - RE-ENABLED for buffering
        self.use_accumulation = True 
        
        # Minimal overlap for smooth transitions (very small to avoid artifacts/buzzing)
        # Reduced overlap to prevent looping/buzzing sounds
        self.overlap_samples = max(32, config.BLOCK_SIZE // 32)  # ~3% overlap, minimum 32 samples
        self.previous_output_tail = None  # Store tail of previous output for crossfading
        
        # Statistics
        self.total_latency = []
        self.skipped_frames = 0  # Track skipped frames due to queue full
        self._last_underflow_warning = 0  # Track last underflow warning time
        
        # Playback control
        self.playback_started = False
        self.min_playback_chunks = 1  # Start playing as soon as we have one processed batch (which is usually large)
        
    def _audio_callback(self, indata, outdata, frames, time_info, status):
        """
        SoundDevice callback.
        1. Reads input audio (indata) -> Puts into input_queue (skip if full to prevent buildup)
        2. Reads processed audio from output_queue -> Writes to output (outdata)
        """
        if status:
            print(f"[Stream Status] {status}", file=sys.stderr)
        
        # Debug: Check if we're receiving audio input (only print occasionally)
        if not hasattr(self, '_input_check_counter'):
            self._input_check_counter = 0
            self._last_input_level = 0.0
            print("[Stream] Audio callback is active and receiving data!")
        
        self._input_check_counter += 1
        if self._input_check_counter == 1:
            print(f"[Stream] First audio callback received! Input shape: {indata.shape}, Output shape: {outdata.shape}")
        
        if self._input_check_counter % 100 == 0:  # Check every 100 frames
            input_level = np.abs(indata).max()
            if input_level > 0.001:  # If there's significant audio
                if abs(input_level - self._last_input_level) > 0.01:  # Only print if level changed
                    print(f"[Stream Debug] Input audio detected: level={input_level:.4f}")
                    self._last_input_level = input_level
            elif self._input_check_counter == 100:  # Print once at start
                print(f"[Stream Warning] No audio input detected (level={input_level:.6f}). Check microphone connection.")
            
        # 1. Handle Input - Always try to put input in queue
        # Use blocking put with very short timeout to ensure we capture audio
        try:
            # Try non-blocking first
            self.input_queue.put_nowait(indata.copy())
            # Track that we're receiving input
            if not hasattr(self, '_first_input_received'):
                self._first_input_received = True
                print(f"[Stream] First audio input received! Queue size: {self.input_queue.qsize()}/{self.input_queue.maxsize}")
        except queue.Full:
            # Queue is full - skip this frame to prevent latency buildup
            self.skipped_frames += 1
            if self.skipped_frames % 100 == 0:  # Print every 100 skipped frames
                print(f"[Stream] Skipped {self.skipped_frames} frames (queue full - processing too slow)")
        
        # 2. Handle Output
        # Check for test tone mode first (for debugging audio output)
        if getattr(self.config, 'TEST_TONE_MODE', False):
            # Generate test tone (440Hz sine wave) to verify audio output works
            samples = len(outdata)
            sample_rate = self.config.SAMPLE_RATE
            t = np.arange(samples) / sample_rate
            tone = 0.3 * np.sin(2 * np.pi * self.test_tone_freq * t + self.test_tone_phase)
            self.test_tone_phase += 2 * np.pi * self.test_tone_freq * samples / sample_rate
            if self.test_tone_phase > 2 * np.pi:
                self.test_tone_phase -= 2 * np.pi
            outdata[:] = tone.reshape(-1, 1)
            if not hasattr(self, '_test_tone_logged'):
                self._test_tone_logged = True
                print(f"[Stream] TEST TONE MODE: Playing 440Hz tone to verify audio output")
            return
            
        # Pre-buffering logic: Don't start playing until we have enough chunks
        # This prevents stuttering at the start
        if not self.playback_started:
            if self.output_queue.qsize() >= self.min_playback_chunks:
                self.playback_started = True
                print(f"[Stream] Pre-buffering complete. Starting playback with {self.output_queue.qsize()} chunks.")
            else:
                # Still buffering - play silence
                outdata[:] = 0
                return
        
        # Always try to get audio
        try:
            # Try to get processed audio from output queue with short timeout
            data = self.output_queue.get(timeout=0.001)  # 1ms timeout - very short
            # Store for repetition if queue becomes empty (though we prefer silence now)
            self.last_output_chunk = data.copy() if isinstance(data, np.ndarray) else data
            
            # Ensure data is the right shape and type
            if isinstance(data, np.ndarray):
                # Ensure it's 2D (samples, channels)
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                elif data.ndim == 2 and data.shape[1] != 1:
                    data = data[:, 0:1]  # Take first channel
                
                # CRITICAL: Ensure dtype matches outdata (float32 for sounddevice)
                if data.dtype != outdata.dtype:
                    data = data.astype(outdata.dtype)
                
                # Ensure data length matches buffer size
                if len(data) < len(outdata):
                    # Pad with zeros if too short
                    outdata[:len(data)] = data
                    outdata[len(data):] = 0
                elif len(data) > len(outdata):
                    # Truncate if too long
                    outdata[:] = data[:len(outdata)]
                else:
                    # Direct assignment - ensure it's a proper copy
                    outdata[:] = data
                
                # Debug: Log first successful output (only once)
                if not hasattr(self, '_first_output_sent'):
                    self._first_output_sent = True
                    output_level = np.abs(data).max()
                    outdata_level = np.abs(outdata).max()
                    print(f"[Stream] First output sent! Data shape: {data.shape}, dtype: {data.dtype}, level: {output_level:.4f}")
                    print(f"[Stream] Outdata shape: {outdata.shape}, dtype: {outdata.dtype}, level after write: {outdata_level:.4f}")
                
                # Debug: Log periodically to ensure output is being written
                if not hasattr(self, '_output_log_counter'):
                    self._output_log_counter = 0
                self._output_log_counter += 1
                if self._output_log_counter % 100 == 0:
                    outdata_level = np.abs(outdata).max()
                    queue_size = self.output_queue.qsize()
                    print(f"[Stream Debug] Output callback: queue_size={queue_size}, outdata_level={outdata_level:.4f}")
            else:
                # Invalid data type - play silence
                outdata[:] = 0
                
        except queue.Empty:
            # Underflow: Model hasn't finished processing yet
            # Play silence instead of repeating chunks to avoid "robotic buzzing"
            outdata[:] = 0
            
            # Reset playback started if queue runs completely dry?
            # Maybe not, as that might cause another long delay.
            # But if it's dry, we ARE delaying anyway.
            
            # Only print warning occasionally to avoid spam
            if not hasattr(self, '_last_underflow_warning') or time.time() - self._last_underflow_warning > 5.0:
                print(f"[Stream Warning] Output queue empty - playing silence (processing too slow)")
                self._last_underflow_warning = time.time()
            
    def _processing_loop(self):
        """
        Runs in a separate thread.
        Takes audio from input_queue -> Runs Model -> Puts into output_queue
        """
        print("[Stream] Processing thread started.")
        print("[Stream] Waiting for audio input...")
        processed_chunks = 0
        failed_chunks = 0
        empty_count = 0
        
        while self.running:
            try:
                # Get input chunk (blocking with timeout to check self.running)
                input_chunk = None
                try:
                    input_chunk = self.input_queue.get(timeout=0.1)
                    empty_count = 0  # Reset counter when we get data
                    self.last_input_time = time.time()
                    if processed_chunks == 0:
                        print(f"[Stream] First chunk received! Starting processing...")
                except queue.Empty:
                    # Queue empty - check for silence/flush condition
                    empty_count += 1
                    if empty_count % 50 == 0:  # Print every 5 seconds (50 * 0.1s)
                        print(f"[Stream] Waiting for audio input... (queue empty, checked {empty_count} times)")
                    
                    # If we have accumulated data and no input for > 1 second, flush it
                    if len(self.chunk_buffer) > 0 and (time.time() - self.last_input_time > 1.0):
                        print(f"[Stream] Silence detected (>1s), flushing accumulated buffer ({len(self.chunk_buffer)} chunks)")
                        # Proceed to process what we have
                        pass
                    else:
                        continue
                
                # Measure processing time
                start_time = time.time()
                
                # If we got a chunk, add to buffer
                if input_chunk is not None:
                    # Flatten input chunk (channels, samples) -> (samples,)
                    input_mono = input_chunk.flatten()
                    
                    # Validate input
                    if input_mono.size == 0:
                        print("[Stream Error] Empty input chunk received")
                        continue
                        
                    self.chunk_buffer.append(input_mono.copy())
                
                # Check if we should process:
                # 1. Enough chunks accumulated
                # 2. Forced flush (silence/timeout)
                # 3. Input queue is effectively empty/silence detected (implicit in logic above)
                
                # Check for silence in current chunk to trigger flush counter
                if input_chunk is not None:
                    input_level = np.abs(input_mono).max()
                    if input_level < 0.01:
                        self.silence_frames += 1
                    else:
                        self.silence_frames = 0
                
                # Conditions to process
                should_process = False
                reason = ""
                
                if len(self.chunk_buffer) >= self.chunks_to_accumulate:
                    should_process = True
                    reason = "buffer full"
                elif len(self.chunk_buffer) > 0 and (time.time() - self.last_input_time > 1.0):
                    should_process = True
                    reason = "timeout flush"
                elif len(self.chunk_buffer) > 0 and self.silence_frames > 43: # Approx 1s of silence
                    should_process = True
                    reason = "silence flush"
                
                if not should_process:
                    # Not enough chunks yet - skip processing this iteration
                    if len(self.chunk_buffer) == 1 and input_chunk is not None:
                        print(f"[Stream] Accumulating chunks: {len(self.chunk_buffer)}/{self.chunks_to_accumulate} (model needs sufficient context to avoid noise)")
                    continue
                
                # Concatenate accumulated chunks into a larger chunk for processing
                accumulated_audio = np.concatenate(self.chunk_buffer)
                self.chunk_buffer = []  # Clear buffer after processing
                self.silence_frames = 0 # Reset silence counter
                
                accumulated_duration_ms = (len(accumulated_audio) / self.config.SAMPLE_RATE) * 1000
                print(f"[Stream] Processing accumulated chunk ({reason}): {len(accumulated_audio)} samples (~{accumulated_duration_ms:.1f}ms) for better quality")
                
                # Use accumulated audio for processing
                input_for_processing = accumulated_audio
                
                # Run Voice Conversion with comprehensive error handling and timing
                inference_start = time.time()
                converted_chunk = None
                
                # Check for pass-through mode
                if getattr(self.config, 'PASSTHROUGH_MODE', False):
                    converted_chunk = input_for_processing.copy()
                else:
                    try:
                        converted_chunk = self.model.process_chunk(input_for_processing)
                        inference_time = (time.time() - inference_start) * 1000
                        
                        # Validate output
                        if converted_chunk is None:
                            print(f"[Stream Error] Model returned None (inference took {inference_time:.1f}ms)")
                            failed_chunks += 1
                            continue
                        
                        if not isinstance(converted_chunk, np.ndarray):
                            print(f"[Stream Error] Model returned invalid type: {type(converted_chunk)} (inference took {inference_time:.1f}ms)")
                            failed_chunks += 1
                            continue
                        
                        if converted_chunk.size == 0:
                            print(f"[Stream Error] Model returned empty output (inference took {inference_time:.1f}ms)")
                            failed_chunks += 1
                            continue
                        
                        # Check for NaN or Inf
                        if np.any(np.isnan(converted_chunk)) or np.any(np.isinf(converted_chunk)):
                            print(f"[Stream Error] Model output contains NaN or Inf values (inference took {inference_time:.1f}ms)")
                            failed_chunks += 1
                            continue
                        
                        # Ensure output is float32 and properly shaped
                        if converted_chunk.dtype != np.float32:
                            converted_chunk = converted_chunk.astype(np.float32)
                        
                        # Ensure it's 1D (samples,) - sounddevice expects 2D but we'll reshape in callback
                        if converted_chunk.ndim > 1:
                            converted_chunk = converted_chunk.flatten()
                        
                        # Apply output gain to boost quiet model output
                        output_gain = getattr(self.config, 'OUTPUT_GAIN', 1.0)
                        if output_gain != 1.0:
                            converted_chunk = converted_chunk * output_gain
                            if processed_chunks == 0:
                                print(f"[Stream] Applied output gain: {output_gain}x")
                        
                        # Normalize to prevent clipping (if needed, after gain)
                        max_val = np.abs(converted_chunk).max()
                        if max_val > 1.0:
                            # Hard normalize to prevent any clipping/distortion
                            converted_chunk = converted_chunk / max_val
                            if processed_chunks % 50 == 0:
                                print(f"[Stream Warning] Output clipped (max={max_val:.4f}), normalized to prevent distortion")
                        elif max_val < 0.01:
                            # If output is very quiet, boost it slightly
                            if processed_chunks % 50 == 0:
                                print(f"[Stream Info] Output is quiet (max={max_val:.4f}), consider increasing OUTPUT_GAIN slightly")
                        
                        # Check if output is all zeros (silence)
                        output_level = np.abs(converted_chunk).max()
                        if np.allclose(converted_chunk, 0, atol=1e-6):
                            if processed_chunks % 50 == 0:  # Print more frequently
                                print(f"[Stream Warning] Model output is all zeros (silence), inference took {inference_time:.1f}ms")
                        elif processed_chunks % 10 == 0:  # Print successful processing occasionally
                            print(f"[Stream] Processed chunk {processed_chunks}: inference={inference_time:.1f}ms, output_level={output_level:.4f}, shape={converted_chunk.shape}, dtype={converted_chunk.dtype}")
                    
                    except Exception as model_error:
                        inference_time = (time.time() - inference_start) * 1000
                        failed_chunks += 1
                        print(f"[Stream Error] Model processing failed after {inference_time:.1f}ms: {type(model_error).__name__}: {model_error}")
                        import traceback
                        traceback.print_exc()
                        # Put silence instead of crashing
                        converted_chunk = np.zeros_like(input_for_processing)
                
                # Split accumulated output back into smaller chunks for real-time output
                # The model processed a large accumulated chunk, but we need to output in smaller pieces
                chunk_size = self.config.BLOCK_SIZE
                overlap = self.overlap_samples
                
                # Split the large converted chunk into smaller output chunks
                output_chunks_to_send = []
                start_idx = 0
                
                while start_idx < len(converted_chunk):
                    end_idx = min(start_idx + chunk_size, len(converted_chunk))
                    output_chunk = converted_chunk[start_idx:end_idx].copy()
                    
                    # Reshape for output (samples, channels)
                    try:
                        if output_chunk.ndim == 1:
                            output_chunk = output_chunk.reshape(-1, 1)
                        else:
                            output_chunk = output_chunk.reshape(-1, 1)
                        
                        # Ensure output is float32 and in valid range [-1, 1]
                        output_chunk = output_chunk.astype(np.float32)
                        output_chunk = np.clip(output_chunk, -1.0, 1.0)
                        
                        # Final validation
                        if output_chunk.size == 0:
                            print("[Stream Error] Output chunk is empty after processing")
                            continue
                        
                        output_chunks_to_send.append(output_chunk)
                        start_idx = end_idx
                    
                    except Exception as reshape_error:
                        print(f"[Stream Error] Failed to reshape output: {reshape_error}")
                        failed_chunks += 1
                        continue
                
                # Send all output chunks to queue
                for output_chunk in output_chunks_to_send:
                    # Blocking put with timeout to ensure audio is NOT discarded
                    try:
                        self.output_queue.put(output_chunk, timeout=1.0)
                        processed_chunks += 1
                    except queue.Full:
                        # Only if queue is completely stuck, we clear old items
                        print(f"[Stream Warning] Output queue full - clearing old audio")
                        while not self.output_queue.empty():
                            try: self.output_queue.get_nowait()
                            except: break
                        self.output_queue.put(output_chunk)
                        processed_chunks += 1

                # -----------------
                # Note: processed_chunks is incremented inside the output loop above
                process_time = (time.time() - start_time) * 1000
                self.total_latency.append(process_time)
                
                # Print diagnostics periodically
                if processed_chunks % 20 == 0:
                    avg_latency = np.mean(self.total_latency[-20:]) if len(self.total_latency) > 0 else 0
                    queue_size = self.output_queue.qsize()
                    print(f"[Stream] Processed batch: Latency={avg_latency:.1f}ms, OutputQueue={queue_size}/{self.output_queue.maxsize}")
                    
            except Exception as e:
                failed_chunks += 1
                print(f"[Stream Error] Unexpected error in processing loop: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                
    def start(self):
        if self.running:
            return

        print(f"[Stream] Starting audio stream (Blocksize: {self.config.BLOCK_SIZE})...")
        self.running = True
        
        # Start Processing Thread
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        # Start Audio Stream
        # Find device IDs from config names (similar to minimal_voice_processing_pipeline)
        try:
            # List available devices for debugging
            devices = sd.query_devices()
            print(f"[Stream] Available audio devices: {len(devices)}")
            
            # List key devices (similar to minimal_voice_processing_pipeline)
            print("[Stream] Key audio devices:")
            for i, device in enumerate(devices):
                # Filter out internal Jetson APE devices to reduce clutter
                if "APE" in device['name']:
                    continue
                
                io_str = []
                if device['max_input_channels'] > 0:
                    io_str.append(f"In: {device['max_input_channels']}")
                if device['max_output_channels'] > 0:
                    io_str.append(f"Out: {device['max_output_channels']}")
                
                # Only show devices that have at least some I/O
                if io_str:
                    io_display = ", ".join(io_str)
                    # Highlight pulse/default devices
                    if 'pulse' in device['name'].lower() or 'default' in device['name'].lower():
                        print(f"  ID {i}: {device['name']} ({io_display}) ⭐ RECOMMENDED")
                    else:
                        print(f"  ID {i}: {device['name']} ({io_display})")
            
            # Helper function to find device by name or use default
            def find_device(device_name, is_input=True):
                """Find device ID by name, or return None for default."""
                if device_name is None:
                    return None
                
                # If it's already an integer, use it
                if isinstance(device_name, int):
                    if 0 <= device_name < len(devices):
                        return device_name
                    else:
                        print(f"[Stream Warning] Invalid device ID {device_name}, using system default")
                        return None
                
                # If it's a string, try to find by name
                if isinstance(device_name, str):
                    device_lower = device_name.lower().strip()
                    
                    # Check for special keywords
                    if device_lower in ['pulse', 'default']:
                        # First, try to find "pulse" device by name (most common on Linux)
                        pulse_found = False
                        for i, dev in enumerate(devices):
                            if 'pulse' in dev['name'].lower():
                                # Verify it has the required channels
                                if is_input and dev['max_input_channels'] > 0:
                                    print(f"[Stream] Found Pulse device: ID {i} - {dev['name']}")
                                    return i
                                elif not is_input and dev['max_output_channels'] > 0:
                                    print(f"[Stream] Found Pulse device: ID {i} - {dev['name']}")
                                    return i
                        
                        # If pulse not found, try "default" device
                        for i, dev in enumerate(devices):
                            if 'default' in dev['name'].lower():
                                if is_input and dev['max_input_channels'] > 0:
                                    print(f"[Stream] Found Default device: ID {i} - {dev['name']}")
                                    return i
                                elif not is_input and dev['max_output_channels'] > 0:
                                    print(f"[Stream] Found Default device: ID {i} - {dev['name']}")
                                    return i
                        
                        # If neither found, use system default (None) - sounddevice will handle it
                        print(f"[Stream] Pulse/Default device not found by name, using system default (None)")
                        return None
                    
                    # Try to find by exact name match or partial match
                    for i, dev in enumerate(devices):
                        if device_name.lower() in dev['name'].lower() or dev['name'].lower() in device_name.lower():
                            # Verify it has the required channels
                            if is_input and dev['max_input_channels'] > 0:
                                print(f"[Stream] Found device: ID {i} - {dev['name']}")
                                return i
                            elif not is_input and dev['max_output_channels'] > 0:
                                print(f"[Stream] Found device: ID {i} - {dev['name']}")
                                return i
                    
                    print(f"[Stream Warning] Device '{device_name}' not found, using system default")
                    return None
                
                return None
            
            # Find input and output devices
            in_dev = find_device(self.config.INPUT_DEVICE, is_input=True)
            out_dev = find_device(self.config.OUTPUT_DEVICE, is_input=False)
            
            print(f"[Stream] Using input device: {in_dev} ({'default' if in_dev is None else devices[in_dev]['name']})")
            print(f"[Stream] Using output device: {out_dev} ({'default' if out_dev is None else devices[out_dev]['name']})")
            
            self.stream = sd.Stream(
                device=(in_dev, out_dev),
                samplerate=self.config.SAMPLE_RATE,
                blocksize=self.config.BLOCK_SIZE,
                channels=1, # Mono
                dtype='float32',  # Explicitly set dtype
                callback=self._audio_callback
            )
            self.stream.start()
            print("[Stream] Audio stream active.")
            print(f"[Stream] Output queue size: {self.output_queue.maxsize}")
            print(f"[Stream] Input queue size: {self.input_queue.maxsize}")
            print(f"[Stream] Waiting for audio input from microphone...")
            print(f"[Stream] Speak into your microphone to start processing!")
            
        except Exception as e:
            print(f"[Stream Error] Failed to start stream: {e}")
            import traceback
            traceback.print_exc()
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
