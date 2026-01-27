# Installation Manual

This guide provides a "Fresh Install" reproduction path for the Voice Conversion project. It covers both high-performance PC setups for the Desktop GUI and optimized edge setups for the NVIDIA Jetson Nano.

---

## 💻 1. PC Installation (Windows / Linux / macOS)

Follow these steps to reproduce the demo on a clean install PC. This setup uses the **best fine-tuned Oxxxymiron model** for high-quality voice conversion.

### Prerequisites
*   **Python 3.10**: (Mandatory for dependency compatibility).
*   **FFmpeg**: 
    *   *Windows*: `choco install ffmpeg` or download binaries.
    *   *Linux*: `sudo apt install ffmpeg`
    *   *macOS*: `brew install ffmpeg`
*   **GPU Drivers**: Latest NVIDIA drivers (for CUDA) or recent macOS (for MPS).

### Reproduction Steps
1.  **Clone & Environment**:
    ```bash
    git clone https://github.com/NikitosKokos/innovation_research_project.git
    cd innovation_research_project
    python -m venv .venv
    
    # Activate:
    .\.venv\Scripts\activate  # Windows
    source .venv/bin/activate # Linux/macOS
    ```

2.  **Install Dependencies**:
    ```bash
    # Windows/Linux (CUDA/CPU):
    pip install -r requirements.txt
    
    # macOS (Apple Silicon):
    pip install -r requirements-mac.txt
    ```

3.  **Run with Fine-tuned Model**:
    To reproduce the Oxxxymiron demo using the provided fine-tuned model:
    ```bash
    python real-time-gui.py --checkpoint-path "AI_models/rapper_oxxxy_finetune/ft_model.pth" --config-path "AI_models/rapper_oxxxy_finetune/config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
    ```

---

## 🟢 2. Jetson Nano Installation (JetPack 6.0)

This setup is specialized for real-time inference on the NVIDIA Jetson Nano edge device.

### Prerequisites
*   **Hardware**: NVIDIA Jetson Nano (4GB/Orin).
*   **OS**: JetPack 6.0 (L4T R36.x).
*   **Dependencies**: CUDA 12.2, cuDNN 9.x.

### Reproduction Steps
1.  **Run Automated Setup**:
    We have provided a comprehensive script that fixes library version symbols and installs PyTorch 2.2.0 (the most stable for cuDNN 9).
    ```bash
    cd innovation_research_project
    chmod +x COMPLETE_SETUP_JETSON_NANO.sh
    ./COMPLETE_SETUP_JETSON_NANO.sh
    ```

2.  **Configure Target Voice**:
    Edit `project/real_time_voice_conversion/config.py` to point `TARGET_VOICE_PATH` to your reference `.wav` file.

3.  **Run Optimized App**:
    ```bash
    python project/real_time_voice_conversion/app.py
    ```

---

## ⚙️ Configuration & Details

### Reproducibility Notes
*   **Cloud Setup**: **None required**. No Azure, AWS, or GCP resources are needed. All processing is local.
*   **API Keys**: **None required**. The system uses open-source models downloaded directly from Hugging Face on the first run.
*   **Environment Variables**: A `.env` file is automatically used by the system to set `OMP_NUM_THREADS=4` for CPU optimization.

### Troubleshooting Clean Installs
*   **ModuleNotFoundError**: Ensure the virtual environment is activated.
*   **CUDA not available**: Verify your PyTorch installation matches your CUDA version (`torch.cuda.is_available()`).
*   **PortAudio Error**: Ensure `libportaudio2` (Linux) or `portaudio` (macOS) is installed via the system package manager.
