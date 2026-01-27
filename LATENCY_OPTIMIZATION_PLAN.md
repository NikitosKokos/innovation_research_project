# Real-Time Voice Conversion Latency Optimization Plan

This guide provides a step-by-step approach to reducing latency from ~10 seconds to 1-2 seconds ("near real-time") on the Jetson Nano. 

## Current Status
- **Model Speed:** Your logs show the model processes ~10 seconds of audio in ~6.4 seconds. This is **0.64x real-time** (faster than real-time).
- **Latency Source:** The main delay is waiting for the 10-second buffer to fill (`CHUNKS_TO_ACCUMULATE = 430`).
- **Goal:** Reduce the buffer size without causing the "stuttering" or "fragmented audio" issues you saw earlier.

---

## Step 1: Reduce Accumulation Buffer (High Impact)
The most direct way to reduce latency is to process smaller chunks of audio. We will reduce the buffer from 10s to ~3s.

1.  Open `project/real_time_voice_conversion/config.py`.
2.  Find `CHUNKS_TO_ACCUMULATE`.
3.  Change it from `430` to **129** (approx 3 seconds).
    ```python
    CHUNKS_TO_ACCUMULATE = 129
    ```
4.  **Test:** Run the app. Speak a sentence.
    *   *Success:* Audio plays back after ~3-4 seconds delay. No stuttering.
    *   *Failure:* Audio stutters or has buzzing gaps.

## Step 2: Reduce Diffusion Steps (Speed Boost)
Fewer steps mean faster processing, allowing us to use even smaller buffers.

1.  Open `project/real_time_voice_conversion/config.py`.
2.  Find `DIFFUSION_STEPS`.
3.  Change it from `10` to **5**.
    ```python
    DIFFUSION_STEPS = 5
    ```
4.  **Test:** Check if the voice quality is still acceptable.
    *   *Note:* Lower steps might make the voice sound slightly "robotic" or less like the target, but it doubles the speed.

## Step 3: Disable CFG (2x Speedup)
Classifier-Free Guidance (CFG) improves quality but usually runs the model twice per step. Disabling it can double performance.

1.  Open `project/real_time_voice_conversion/config.py`.
2.  Find `INFERENCE_CFG_RATE`.
3.  Change it from `0.7` to **0.0**.
    ```python
    INFERENCE_CFG_RATE = 0.0
    ```
4.  **Test:** This should make processing extremely fast.
    *   *Check:* Does the voice still sound like the target? If yes, keep this.

## Step 4: "Near Real-Time" Buffer (Target Goal)
If Steps 2 & 3 made processing fast enough, we can now aim for 1-second latency.

1.  Open `project/real_time_voice_conversion/config.py`.
2.  Change `CHUNKS_TO_ACCUMULATE` to **43** (approx 1 second).
    ```python
    CHUNKS_TO_ACCUMULATE = 43
    ```
3.  **Test:** Speak continuously.
    *   *Goal:* You should hear your voice back with only ~1-1.5s delay.

## Step 5: System Verification (Debugging)
If you encounter issues, verify your environment versions to ensure no software bottlenecks. Run this command in your terminal:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, CuDNN: {torch.backends.cudnn.version()}')"
```

*   **PyTorch:** Should be 2.1.0 or newer.
*   **CUDA:** Should be 11.4 or 12.x (JetPack standard).
*   **CuDNN:** Should be 8.x.

## Step 6: Advanced - TensorRT (Future Work)
If the Python-only optimizations above don't yield perfect 1s latency, the next step is compiling the model to TensorRT. This is complex and requires:
1.  Exporting the PyTorch model to ONNX.
2.  Converting ONNX to TensorRT Engine (`.trt`).
3.  Updating the inference code to use the TensorRT engine.

*Note: Try Steps 1-4 first, as they are likely sufficient given your logs showing 0.64x real-time performance.*
