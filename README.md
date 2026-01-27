# Innovation and Research Project: Voice Conversion

This project implements a state-of-the-art Voice Conversion (VC) system based on **Seed-VC**. The primary goal is to achieve **real-time voice conversion on the NVIDIA Jetson Nano** edge device, converting a source speaker's voice into that of a target speaker (e.g., the Russian rapper Oxxxymiron) while maintaining low latency and high quality.

The project supports zero-shot voice conversion, meaning you can clone a voice with just 1-30 seconds of reference audio. It is optimized for both edge deployment and standard PC testing.

## 📚 Documentation

- **[User Manual](USER_MANUAL.md)**: A practical guide on how to use the application, including the **Real-time GUI for PC** and the **Optimized App for Jetson Nano**.
- **[Installation Manual](INSTALLATION.md)**: Detailed "Fresh Install" instructions for Windows, Linux, macOS, and Jetson Nano (JetPack 6.0).
- **[Model Upload Guide](UPLOAD_MODELS_GUIDE.md)**: Instructions for handling large AI models (>100MB) using Git LFS.
- **[Appendices](APPENDICES.md)**: Technical diagrams (architecture, data flow), visual mockups, and performance visualizations.

## 🚀 Quick Start (PC)

To run the real-time demo on a PC with the best fine-tuned model:

1.  **Setup Environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # (or .\venv\Scripts\activate on Windows)
    pip install -r requirements.txt
    ```

2.  **Run the GUI**:
    ```bash
    python real-time-gui.py --checkpoint-path "AI_models/rapper_oxxxy_finetune/ft_model.pth" --config-path "AI_models/rapper_oxxxy_finetune/config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
    ```

## ✨ Key Features

-   **Jetson Nano Optimization**: Specifically tuned for real-time inference on edge hardware using specialized buffering and model pruning.
-   **Oxxxymiron Fine-tuned Model**: Includes a high-quality model specifically trained on the voice of the artist [Oxxxymiron](https://en.wikipedia.org/wiki/Oxxxymiron).
-   **Zero-Shot Conversion**: Clone any voice instantly from a short audio sample without further training.
-   **Multiple Interfaces**: Choose between a Desktop GUI (Real-time), Web UI (File processing), or Console App (Jetson optimized).

## 🧠 Model Architecture

The system uses a 3-stage pipeline:
1.  **Speech Tokenizer**: Extracts linguistic content (Whisper/XLSR).
2.  **Diffusion Transformer (DiT)**: Generates acoustic features for the target voice.
3.  **Vocoder**: Reconstructs high-fidelity audio (BigVGAN/HiFi-GAN).

## 🟢 Jetson Nano Performance
*   **Latency**: ~5.0s end-to-end.
*   **Precision**: FP16 hardware acceleration.
*   **Optimizations**: Layer skipping, TensorRT support, and reduced diffusion steps.

## 🔗 Credits
Based on the [Seed-VC](https://github.com/Plachtaa/seed-vc) research project and subsequent optimizations for edge computing.
