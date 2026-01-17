# Seed-VC Streamlit Educational Demo

## Overview

This Streamlit application provides an interactive, educational demonstration of the Seed-VC voice conversion pipeline. It's designed to explain how the system works step-by-step, making it perfect for video production and educational purposes.

## Features

-  **📚 Comprehensive Explanations**: Step-by-step breakdown of the voice conversion pipeline
-  **🎯 Terminology Guide**: Detailed explanations of key concepts (zero-shot, diffusion, mel spectrograms, etc.)
-  **🎵 Interactive Demo**: Try voice conversion with your own audio files
-  **📊 Visualizations**: Waveforms, spectrograms, and F0 contours
-  **🔬 Technical Details**: Deep dive into each processing step

## Installation

1. Make sure you have the main Seed-VC dependencies installed:

```bash
pip install -r requirements.txt
```

2. The Streamlit demo requires additional dependencies (already in requirements.txt):
   -  streamlit
   -  matplotlib

## Running the Demo

```bash
source ../.venv/Scripts/activate
streamlit run streamlit_demo.py
```

The app will open in your browser at `http://localhost:8501`

## Structure

The demo is organized into 5 main sections:

### 1. 🏠 Overview

-  Project introduction
-  Key features
-  Use cases
-  Model statistics

### 2. 🔬 How It Works

Detailed explanation of each pipeline step:

-  Audio Input & Preprocessing
-  Semantic Content Extraction (Whisper/XLSR)
-  Speaker Embedding Extraction (CAMPPlus)
-  Mel Spectrogram Conversion
-  F0 (Pitch) Extraction
-  Length Regulation
-  Diffusion Model (DiT)
-  Vocoder (BigVGAN/HiFT)
-  Post-processing

### 3. 🎯 Terminology

Explanations of key terms with analogies:

-  Zero-Shot Voice Conversion
-  One-Shot / Few-Shot Learning
-  Diffusion Model
-  Mel Spectrogram
-  Speaker Embedding
-  Semantic Content
-  F0 (Fundamental Frequency)
-  Vocoder
-  CFG (Classifier-Free Guidance)
-  Timbre
-  Real-Time Conversion
-  Chunking

### 4. 🎵 Try It Yourself

Interactive voice conversion:

-  Upload source and reference audio
-  Adjust conversion parameters
-  View real-time progress
-  See audio visualizations

### 5. 📊 Results

-  Audio comparison (source, reference, converted)
-  Visual comparisons (waveforms, spectrograms, F0)
-  Analysis guidelines

## For Video Production

This demo is designed to be screen-recorded for educational videos. The structure follows a logical flow:

1. **Introduction (Overview)** - 1 minute
2. **Technical Explanation (How It Works)** - 3 minutes
3. **Terminology (Key Concepts)** - 1 minute
4. **Live Demo (Try It Yourself)** - 1 minute
5. **Results & Analysis** - 30 seconds

**Total: ~6-7 minutes**

## Tips for Recording

1. Start with the Overview section to set context
2. Use the "How It Works" section to explain the pipeline step-by-step
3. Pause on each step to explain the technical details
4. Use the Terminology section to clarify any confusing terms
5. Show a live conversion in "Try It Yourself"
6. End with Results to show the final output

## Customization

You can customize the demo by:

-  Modifying the CSS styles in the `st.markdown()` sections
-  Adding more example audio files
-  Adjusting the visualization parameters
-  Adding more terminology explanations
-  Including additional analysis metrics

## Requirements

-  Python 3.10+
-  All Seed-VC dependencies (see requirements.txt)
-  GPU recommended for actual conversion (CPU will work but be slow)

## Notes

-  The demo uses the existing `scripts/convert_voice.py` for actual conversion
-  Audio files are temporarily saved during processing
-  Visualizations use matplotlib for plotting
-  The app is designed to be educational, not production-ready

## Troubleshooting

If you encounter issues:

1. Make sure all dependencies are installed
2. Check that example audio files exist in `examples/` directory
3. Ensure you have write permissions for temporary files
4. For GPU issues, check CUDA installation

## License

Same as Seed-VC project (GPL-3.0)
