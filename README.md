# Innovation and Research Project: Voice Conversion

## Project Overview
This project focuses on **Voice Conversion (VC)**, a technology that converts the voice of a source speaker into that of a target speaker of russian hip-hop artist [Oxxxymiron](https://en.wikipedia.org/wiki/Oxxxymiron) while preserving the linguistic content and prosody. 

I utilize the **Seed-VC** open-source models for this implementation. Seed-VC is a state-of-the-art voice conversion system capable of zero-shot voice conversion, meaning it can clone a voice without requiring training data for that specific speaker, all it takes is a sample of someone's voice (1-30 seconds), and it copies person's tembre, style and copies his voice.

## Setup Instructions

Follow these steps to set up the development environment.

### 1. Create a Virtual Environment (`.venv`)

It is recommended to use a virtual environment to manage dependencies.

**Windows:**
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

Once the virtual environment is activated, install the required packages.

**Windows and Linux:**
```bash
pip install -r requirements.txt
```

**macOS (Apple Silicon M-Series):**
```bash
pip install -r requirements-mac.txt
```

## Running the Application

This project provides several interfaces for voice conversion.

**Integrated Web UI:**
```bash
python app.py --enable-v1 --enable-v2
```

**Real-time GUI:**
```bash
python real-time-gui.py
```

For more detailed usage, training instructions, and model descriptions, please refer to the original [Seed-VC Documentation](README_SEED_VC.md).
