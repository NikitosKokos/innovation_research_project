# Lite Preprocessing Explained

## Overview

The `LitePreprocessor` is a lightweight audio preprocessing system designed for real-time voice conversion on edge devices like Jetson Nano. It performs two main operations with minimal latency (<5ms).

## Components

### 1. High-Pass Filter (HPF)

**Purpose**: Removes low-frequency noise and rumble (wind, mechanical vibrations, etc.)

**How it works**:
- Uses a **Butterworth 2nd-order high-pass filter**
- Cutoff frequency: **80 Hz** (default)
- Filters out frequencies below 80 Hz while preserving speech (typically 85-8000 Hz)

**Implementation**:
```python
# Design filter
nyquist = 0.5 * sample_rate  # Half the sample rate (max frequency)
norm_cutoff = high_pass_freq / nyquist  # Normalize cutoff frequency
self.b, self.a = scipy.signal.butter(2, norm_cutoff, btype='high')
```

**Streaming State**:
- Uses `lfilter_zi` to maintain filter state between chunks
- This ensures smooth filtering across audio chunks without discontinuities
- The state (`self.zi`) is preserved between calls

**Why it's "lite"**:
- Simple 2nd-order filter (fast computation)
- Streaming implementation (no need to buffer entire audio)
- Low memory footprint

### 2. Noise Gate

**Purpose**: Mutes audio when it's too quiet (background noise, silence)

**How it works**:
- Calculates **RMS (Root Mean Square)** of the audio chunk
- Compares RMS to a threshold (default: -45 dB)
- If below threshold → mute (return zeros)
- If above threshold → pass through

**Implementation**:
```python
# Calculate RMS
rms = np.sqrt(np.mean(filtered_chunk**2))

# Compare to threshold (converted from dB to linear)
if rms < self.noise_gate_threshold:
    return np.zeros_like(filtered_chunk)  # Mute
else:
    return filtered_chunk  # Pass through
```

**Why "hard gate"**:
- No attack/release envelope (zero latency)
- Instant on/off (fastest possible)
- Trade-off: May cause slight clicks, but acceptable for real-time

**Threshold**:
- Default: **-45 dB** (very quiet)
- Lower = more sensitive (mutes more)
- Higher = less sensitive (mutes less)

## Processing Flow

```
Input Audio Chunk
    ↓
[High-Pass Filter] → Removes <80 Hz noise
    ↓
[Noise Gate] → Mutes if too quiet
    ↓
Output Audio Chunk
```

## Usage in Real-Time Pipeline

In `stream.py`, the preprocessor is used in the processing loop:

```python
# 1. Get input chunk from microphone
input_chunk = self.input_queue.get()

# 2. Flatten to mono
input_mono = input_chunk.flatten()

# 3. Apply lite preprocessing
input_mono = self.preprocessor.process(input_mono)

# 4. Run voice conversion model
converted_chunk = self.model.process_chunk(input_mono)
```

## Benefits for Edge Devices

1. **Low Latency**: <5ms processing time
2. **CPU Efficient**: Simple operations (filtering, RMS calculation)
3. **Memory Efficient**: No buffering, processes chunk-by-chunk
4. **Quality Improvement**: 
   - Removes low-frequency noise that can confuse the model
   - Reduces processing of silence/background noise

## Configuration

You can adjust the preprocessing in `stream.py`:

```python
self.preprocessor = LitePreprocessor(
    sample_rate=config.SAMPLE_RATE,
    noise_gate_db=-45.0,  # Adjust: -50 = more sensitive, -40 = less sensitive
    high_pass_freq=80.0   # Adjust: 60 = less filtering, 100 = more filtering
)
```

## Trade-offs

**Pros**:
- Fast and lightweight
- Improves audio quality
- Reduces unnecessary processing

**Cons**:
- Hard gate may cause slight clicks (acceptable for real-time)
- Fixed threshold (not adaptive)
- Simple filtering (not as sophisticated as full preprocessing)

## When to Use

- ✅ Real-time voice conversion
- ✅ Edge devices (Jetson Nano)
- ✅ Low-latency requirements
- ✅ When you need basic noise reduction

## When NOT to Use

- ❌ Offline processing (use full preprocessing instead)
- ❌ When you need advanced noise reduction
- ❌ When you need adaptive thresholds
