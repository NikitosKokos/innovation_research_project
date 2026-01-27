# User Manual

The primary objective of this project is to achieve **real-time voice conversion on the NVIDIA Jetson Nano** edge device. However, for testing, high-quality rendering, and ease of use, a feature-rich **Desktop GUI** is also provided for standard PCs.

---

### 1. Desktop GUI (PC - Real-Time Microphone)

The Desktop GUI is the easiest way to interact with the system on a Windows, Linux, or macOS machine.

**How to Start:**
To use the best fine-tuned model (Oxxxymiron rapper model), run the following command in your terminal:
```bash
python real-time-gui.py --checkpoint-path "AI_models/rapper_oxxxy_finetune/ft_model.pth" --config-path "AI_models/rapper_oxxxy_finetune/config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
```

**How to Use:**
1.  **Load Reference**: Click "Browse" and select the target voice `.wav` file (e.g., a sample of Oxxxymiron's voice from `audio_inputs/reference/`).
2.  **Select Devices**:
    *   **Input Device**: Select your microphone.
    *   **Output Device**: Select your headphones (**mandatory to prevent feedback**).
3.  **Adjust Settings**:
    *   **Block time**: Set to `0.3s` - `0.5s` for low latency.
    *   **Diffusion steps**: Set to `6` - `10` for a balance of speed and quality.
4.  **Start**: Click **Start Voice Conversion** and speak into the mic.

---

### 2. Jetson Nano Console (Optimized Edge App)

This is the core implementation designed for the Jetson Nano. It uses a lightweight console interface to maximize resources for AI inference.

**How to Start:**
1.  Navigate to the project root on your Jetson.
2.  Run the optimized application:
    ```bash
    python project/real_time_voice_conversion/app.py
    ```

**How to Use:**
1.  **Initialization**: Wait for the message "Ready to Start!". The model will pre-calculate the target style from the voice defined in `config.py`.
2.  **Streaming**: 
    *   Press `s` and then **Enter** to start the microphone stream.
    *   Press **Enter** again to pause/stop the stream.
3.  **Quit**: Press `q` and **Enter** to exit.

---

### 3. Web Interface (File Processing)

Best for converting existing audio files or songs with the highest possible quality.

**How to Start:**
```bash
python app.py --enable-v1 --enable-v2
```
Open the local URL (e.g., `http://127.0.0.1:7860`) in your browser.

**Features:**
*   **V1 Tab**: Best for singing covers (preserves melody).
*   **V2 Tab**: Best for speech (preserves emotion and accent).
*   **Style Transfer**: Check "convert style/emotion/accent" in V2 for more expressive results.

---

### 💡 Pro Tips for Senior Results
*   **Microphone**: Use a clear mic without background noise.
*   **Audio group**: If you get "Permission Denied" for audio on Linux/Jetson, run `sudo usermod -aG audio $USER` and log out/in.
*   **Latency**: On Jetson Nano, end-to-end latency is optimized to ~5s. If you experience stuttering, increase the `BLOCK_SIZE` in `project/real_time_voice_conversion/config.py`.
