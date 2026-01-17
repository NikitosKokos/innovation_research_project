"""
Seed-VC Educational Pipeline Demo
A comprehensive Streamlit app explaining how Seed-VC works for video production
"""

import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Page configuration
st.set_page_config(
    page_title="Seed-VC Pipeline Demo",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .term-definition {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .pipeline-step {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversion_complete' not in st.session_state:
    st.session_state.conversion_complete = False
if 'source_audio' not in st.session_state:
    st.session_state.source_audio = None
if 'reference_audio' not in st.session_state:
    st.session_state.reference_audio = None
if 'converted_audio' not in st.session_state:
    st.session_state.converted_audio = None

def plot_waveform(audio, sr, title="Waveform", max_duration=10):
    """Plot audio waveform"""
    duration = len(audio) / sr
    if duration > max_duration:
        # Show only first max_duration seconds
        audio = audio[:int(max_duration * sr)]
        duration = max_duration
    
    fig, ax = plt.subplots(figsize=(12, 3))
    time_axis = np.linspace(0, duration, len(audio))
    ax.plot(time_axis, audio, linewidth=0.5, color='#667eea')
    ax.set_xlabel('Time (seconds)', fontsize=10)
    ax.set_ylabel('Amplitude', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration)
    plt.tight_layout()
    return fig

def plot_spectrogram(audio, sr, title="Mel Spectrogram", n_fft=2048, hop_length=512, n_mels=80):
    """Plot mel spectrogram"""
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, 
        hop_length=hop_length, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    img = ax.imshow(
        mel_db, 
        aspect='auto', 
        origin='lower',
        cmap='viridis',
        interpolation='nearest'
    )
    ax.set_xlabel('Time (frames)', fontsize=10)
    ax.set_ylabel('Mel Frequency Bins', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Convert frames to time
    time_axis = librosa.frames_to_time(np.arange(mel_db.shape[1]), sr=sr, hop_length=hop_length)
    frame_indices = np.linspace(0, len(time_axis)-1, 5).astype(int)
    ax.set_xticks(frame_indices)
    ax.set_xticklabels([f'{time_axis[i]:.1f}s' for i in frame_indices])
    
    plt.colorbar(img, ax=ax, label='dB')
    plt.tight_layout()
    return fig

def plot_f0_contour(audio, sr, title="F0 (Pitch) Contour"):
    """Plot F0 contour using librosa"""
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7')
        )
        
        fig, ax = plt.subplots(figsize=(12, 4))
        times = librosa.times_like(f0, sr=sr)
        
        ax.plot(times, f0, linewidth=2, color='#e74c3c', label='F0')
        ax.fill_between(times, f0, alpha=0.3, color='#e74c3c')
        ax.set_xlabel('Time (seconds)', fontsize=10)
        ax.set_ylabel('Frequency (Hz)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(50, 500)  # Typical human voice range
        plt.tight_layout()
        return fig
    except:
        # Fallback if pyin fails
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, 'F0 extraction not available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=12, fontweight='bold')
        return fig

def load_audio_file(uploaded_file):
    """Load audio from uploaded file"""
    try:
        # Save to temporary file
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load with librosa
        audio, sr = librosa.load(temp_path, sr=None)
        os.remove(temp_path)
        return audio, sr
    except Exception as e:
        st.error(f"Error loading audio: {str(e)}")
        return None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🎙️ Seed-VC Voice Conversion Pipeline</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem;'>Zero-Shot Voice Conversion with Diffusion Models</p>
        <p>Transform any voice into another using just a short reference audio</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("📚 Navigation")
    page = st.sidebar.radio(
        "Choose a section:",
        ["🏠 Overview", "🔬 How It Works", "🎯 Terminology", "🎵 Try It Yourself", "📊 Results"]
    )
    
    if page == "🏠 Overview":
        show_overview()
    elif page == "🔬 How It Works":
        show_pipeline()
    elif page == "🎯 Terminology":
        show_terminology()
    elif page == "🎵 Try It Yourself":
        show_conversion_demo()
    elif page == "📊 Results":
        show_results()

def show_overview():
    """Show project overview"""
    st.markdown('<h2 class="section-header">Project Overview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h3>What is Seed-VC?</h3>
            <p>Seed-VC is a state-of-the-art <strong>zero-shot voice conversion</strong> system that can 
            transform any voice into another voice using just a short reference audio (1-30 seconds). 
            It uses advanced deep learning techniques including diffusion models to achieve high-quality 
            voice conversion without requiring training on the target speaker.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h3>Key Features</h3>
            <ul>
                <li>🎯 <strong>Zero-Shot Conversion:</strong> No training needed for new voices</li>
                <li>⚡ <strong>Real-Time Support:</strong> ~300ms algorithm delay for live conversion</li>
                <li>🎶 <strong>Singing Voice Conversion:</strong> Preserves pitch and musicality</li>
                <li>🔊 <strong>High Quality:</strong> Uses BigVGAN vocoder for natural sound</li>
                <li>📱 <strong>One-Shot Learning:</strong> Works with just 1 utterance per speaker</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 1.5rem; border-radius: 0.5rem;">
            <h3>📈 Model Statistics</h3>
            <p><strong>Model Size:</strong> 25M - 200M params</p>
            <p><strong>Sampling Rate:</strong> 22kHz / 44kHz</p>
            <p><strong>Latency:</strong> ~300ms (real-time)</p>
            <p><strong>Reference Length:</strong> 1-30 seconds</p>
            <p><strong>Supported Formats:</strong> WAV, FLAC, MP3</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>Use Cases</h3>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem;">
            <div style="padding: 1rem; background: white; border-radius: 0.5rem;">
                <strong>🎬 Content Creation</strong><br>
                Dubbing, voiceovers, character voices
            </div>
            <div style="padding: 1rem; background: white; border-radius: 0.5rem;">
                <strong>📢 Live Performance</strong><br>
                Real-time voice transformation
            </div>
            <div style="padding: 1rem; background: white; border-radius: 0.5rem;">
                <strong>🎤 Music Production</strong><br>
                Singing voice conversion, covers
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_pipeline():
    """Show the voice conversion pipeline step-by-step"""
    st.markdown('<h2 class="section-header">🔬 Voice Conversion Pipeline</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <p>The voice conversion process involves several sophisticated steps. Let's explore each one:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 1: Audio Input
    st.markdown("""
    <div class="pipeline-step">
        <h3>📥 Step 1: Audio Input & Preprocessing</h3>
        <p><strong>What happens:</strong> The source audio (voice to convert) and reference audio (target voice) 
        are loaded and preprocessed.</p>
        <ul>
            <li>Audio is resampled to appropriate sample rates (16kHz for semantic extraction, 22kHz/44kHz for conversion)</li>
            <li>Silence is trimmed from the beginning and end</li>
            <li>Volume is normalized to prevent clipping</li>
            <li>Optional noise reduction is applied</li>
        </ul>
        <p><strong>Why it matters:</strong> Clean input audio ensures better conversion quality and reduces artifacts.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 2: Semantic Extraction
    st.markdown("""
    <div class="pipeline-step">
        <h3>🧠 Step 2: Semantic Content Extraction</h3>
        <p><strong>What happens:</strong> The linguistic content (what is being said) is extracted from both audios 
        using a speech encoder.</p>
        <ul>
            <li><strong>Whisper/XLSR Model:</strong> Extracts semantic features that represent the linguistic content</li>
            <li>These features capture <em>what</em> is being said, not <em>how</em> it's said</li>
            <li>For long audio (>30s), the audio is chunked with overlapping windows</li>
        </ul>
        <p><strong>Why it matters:</strong> This separates the content (words, meaning) from the speaker characteristics 
        (voice timbre, accent, emotion).</p>
        <p><strong>Technical detail:</strong> Uses OpenAI Whisper encoder or XLSR (Wav2Vec2) to extract 768-dimensional 
        semantic embeddings at 50Hz frame rate.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 3: Speaker Embedding
    st.markdown("""
    <div class="pipeline-step">
        <h3>👤 Step 3: Speaker Embedding Extraction</h3>
        <p><strong>What happens:</strong> The reference audio is analyzed to extract speaker characteristics.</p>
        <ul>
            <li><strong>CAMPPlus Model:</strong> Extracts 192-dimensional speaker embeddings</li>
            <li>These embeddings capture voice timbre, accent, speaking style, and emotional tone</li>
            <li>Only extracted from the reference audio (target voice)</li>
        </ul>
        <p><strong>Why it matters:</strong> This embedding represents the "voice identity" we want to transfer to 
        the source audio.</p>
        <p><strong>Technical detail:</strong> Uses CAMPPlus (a DTDNN-based speaker verification model) to extract 
        speaker-discriminative features from mel-frequency cepstral coefficients (MFCCs).</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 4: Mel Spectrogram
    st.markdown("""
    <div class="pipeline-step">
        <h3>📊 Step 4: Mel Spectrogram Conversion</h3>
        <p><strong>What happens:</strong> Audio waveforms are converted to mel spectrograms - a frequency-time representation.</p>
        <ul>
            <li><strong>STFT (Short-Time Fourier Transform):</strong> Converts time-domain audio to frequency domain</li>
            <li><strong>Mel Scale:</strong> Applies mel filterbank to match human auditory perception</li>
            <li>Results in a 2D representation: frequency (mel bins) vs. time (frames)</li>
        </ul>
        <p><strong>Why it matters:</strong> Mel spectrograms are easier for neural networks to process than raw audio, 
        and they preserve important acoustic information while being more compact.</p>
        <p><strong>Technical detail:</strong> Uses 80 mel bins, 2048 FFT size, 512 hop length. The mel scale is 
        logarithmic, matching how humans perceive pitch differences.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Visual example of mel spectrogram
    st.markdown("#### 📈 Mel Spectrogram Visualization")
    example_audio_path = "examples/source/source_s1.wav"
    if os.path.exists(example_audio_path):
        audio, sr = librosa.load(example_audio_path, sr=22050, duration=3.0)
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_waveform(audio, sr, "Original Waveform"))
        with col2:
            st.pyplot(plot_spectrogram(audio, sr, "Mel Spectrogram"))
    
    # Step 5: F0 Extraction (optional)
    st.markdown("""
    <div class="pipeline-step">
        <h3>🎵 Step 5: F0 (Pitch) Extraction (Optional)</h3>
        <p><strong>What happens:</strong> For singing voice conversion, the fundamental frequency (F0/pitch) is extracted.</p>
        <ul>
            <li><strong>RMVPE Model:</strong> Robust pitch estimation that works even with background music</li>
            <li>F0 represents the pitch contour of the voice over time</li>
            <li>Can be adjusted (auto-adjust or semitone shift) to match target pitch range</li>
        </ul>
        <p><strong>Why it matters:</strong> Preserves the musical pitch information, essential for singing voice conversion.</p>
        <p><strong>When used:</strong> Only for singing voice conversion (f0_condition=True). For regular speech, 
        this step is skipped.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 6: Length Regulation
    st.markdown("""
    <div class="pipeline-step">
        <h3>⏱️ Step 6: Length Regulation & Conditioning</h3>
        <p><strong>What happens:</strong> The semantic features are processed and conditioned for the diffusion model.</p>
        <ul>
            <li><strong>Length Regulator:</strong> Adjusts the length of semantic features to match desired output length</li>
            <li>Combines semantic content with F0 information (if available)</li>
            <li>Prepares conditioning information for the diffusion process</li>
        </ul>
        <p><strong>Why it matters:</strong> Ensures the output audio has the correct duration and timing.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 7: Diffusion Model
    st.markdown("""
    <div class="pipeline-step">
        <h3>🎨 Step 7: Diffusion Model - Voice Conversion</h3>
        <p><strong>What happens:</strong> This is the core of the voice conversion - a diffusion model generates 
        the target mel spectrogram.</p>
        <ul>
            <li><strong>DiT (Diffusion Transformer):</strong> A transformer-based diffusion model</li>
            <li>Starts with noise and iteratively denoises it</li>
            <li>Conditioned on:
                <ul>
                    <li>Semantic content from source audio</li>
                    <li>Speaker embedding from reference audio</li>
                    <li>F0 contour (if singing voice conversion)</li>
                </ul>
            </li>
            <li>Uses <strong>CFG (Classifier-Free Guidance)</strong> to control the balance between content preservation 
            and voice similarity</li>
        </ul>
        <p><strong>Why it matters:</strong> The diffusion process allows for high-quality, controllable generation 
        of the target voice while preserving the linguistic content.</p>
        <p><strong>Technical detail:</strong> Uses 25-50 diffusion steps. More steps = better quality but slower. 
        CFG rate (0.0-1.0) controls how strongly the model follows the conditioning.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 8: Vocoder
    st.markdown("""
    <div class="pipeline-step">
        <h3>🔊 Step 8: Vocoder - Waveform Generation</h3>
        <p><strong>What happens:</strong> The generated mel spectrogram is converted back to audio waveform.</p>
        <ul>
            <li><strong>BigVGAN or HiFT Vocoder:</strong> Neural vocoder that synthesizes audio from mel spectrograms</li>
            <li>BigVGAN is used for high-quality singing voice (44kHz)</li>
            <li>HiFT is used for real-time voice conversion (22kHz)</li>
            <li>Generates natural-sounding audio with proper phase information</li>
        </ul>
        <p><strong>Why it matters:</strong> The vocoder is crucial for final audio quality. It reconstructs the 
        time-domain waveform from the frequency representation.</p>
        <p><strong>Technical detail:</strong> BigVGAN uses generator networks with anti-aliasing for high-fidelity 
        audio generation. HiFT is optimized for real-time inference.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 9: Post-processing
    st.markdown("""
    <div class="pipeline-step">
        <h3>✨ Step 9: Post-processing & Output</h3>
        <p><strong>What happens:</strong> Final audio is processed and saved.</p>
        <ul>
            <li>For long audio: chunks are crossfaded together smoothly</li>
            <li>Audio is normalized to prevent clipping</li>
            <li>Output is saved in the desired format (WAV, etc.)</li>
        </ul>
        <p><strong>Why it matters:</strong> Ensures smooth transitions between chunks and proper audio levels.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Pipeline diagram
    st.markdown("""
    <div class="info-box">
        <h3>📋 Complete Pipeline Summary</h3>
        <pre style="background: white; padding: 1rem; border-radius: 0.5rem; overflow-x: auto;">
Source Audio → Preprocessing → Semantic Extraction (Whisper/XLSR)
                                                      ↓
Reference Audio → Preprocessing → Speaker Embedding (CAMPPlus)
                                                      ↓
                    Mel Spectrogram Conversion
                                                      ↓
                    F0 Extraction (optional)
                                                      ↓
                    Length Regulation
                                                      ↓
                    Diffusion Model (DiT) ← Conditioning
                                                      ↓
                    Vocoder (BigVGAN/HiFT)
                                                      ↓
                    Post-processing
                                                      ↓
                    Converted Audio Output
        </pre>
    </div>
    """, unsafe_allow_html=True)

def show_terminology():
    """Show terminology explanations"""
    st.markdown('<h2 class="section-header">🎯 Key Terminology Explained</h2>', unsafe_allow_html=True)
    
    terms = [
        {
            "term": "Zero-Shot Voice Conversion",
            "definition": "The ability to convert a voice to a new speaker without any training data from that speaker. The model generalizes from its training to work with unseen voices.",
            "analogy": "Like a translator who can translate between any two languages they know, even if they've never seen that specific combination before."
        },
        {
            "term": "One-Shot / Few-Shot Learning",
            "definition": "Learning to perform a task with just 1 (one-shot) or a few (few-shot) examples. In voice conversion, this means the model can clone a voice using just 1-30 seconds of reference audio.",
            "analogy": "Like learning to imitate someone's voice after hearing them speak just once."
        },
        {
            "term": "Diffusion Model",
            "definition": "A generative AI model that creates data by iteratively removing noise. It starts with random noise and gradually refines it into the target output (like a mel spectrogram) through multiple steps.",
            "analogy": "Like a sculptor who starts with a block of marble (noise) and gradually carves away to reveal the statue (target voice)."
        },
        {
            "term": "Mel Spectrogram",
            "definition": "A 2D representation of audio showing frequency content over time, using the mel scale which matches human auditory perception. It's like a 'picture' of sound.",
            "analogy": "Like a musical score, but instead of notes, it shows how different frequencies (pitches) change over time."
        },
        {
            "term": "Speaker Embedding",
            "definition": "A numerical vector (192 dimensions in Seed-VC) that captures the unique characteristics of a speaker's voice: timbre, accent, speaking style, and emotional tone.",
            "analogy": "Like a DNA fingerprint, but for voice - a unique identifier that captures what makes a voice sound like that person."
        },
        {
            "term": "Semantic Content",
            "definition": "The linguistic meaning of speech - what words are being said and their meaning, separate from how they're said (voice characteristics).",
            "analogy": "The text/meaning of what someone says, independent of their voice. Like the difference between the words 'Hello' and the way different people say it."
        },
        {
            "term": "F0 (Fundamental Frequency)",
            "definition": "The pitch of the voice - the lowest frequency component that determines how high or low a voice sounds. Measured in Hz (Hertz).",
            "analogy": "Like the note on a piano - F0 determines if a voice sounds high-pitched (soprano) or low-pitched (bass)."
        },
        {
            "term": "Vocoder",
            "definition": "A neural network that converts mel spectrograms (frequency representation) back into audio waveforms (actual sound). It's the final step that creates the audible output.",
            "analogy": "Like a speaker that converts electrical signals into sound - but in this case, it converts frequency data into actual audio."
        },
        {
            "term": "CFG (Classifier-Free Guidance)",
            "definition": "A technique that controls how strongly the model follows the conditioning information (like speaker embedding). Higher CFG = stronger voice similarity, lower CFG = more content preservation.",
            "analogy": "Like a volume knob - turning it up makes the output voice more similar to the reference, turning it down preserves more of the original characteristics."
        },
        {
            "term": "Timbre",
            "definition": "The unique quality or 'color' of a voice that makes it distinguishable from others, even when saying the same thing at the same pitch. It's what makes voices sound different.",
            "analogy": "Like the difference between a violin and a piano playing the same note - same pitch, different sound quality."
        },
        {
            "term": "Real-Time Conversion",
            "definition": "Voice conversion that happens fast enough for live use (like in video calls or gaming), with latency under ~400ms total.",
            "analogy": "Like live translation - the conversion happens as you speak, with minimal delay."
        },
        {
            "term": "Chunking",
            "definition": "Breaking long audio into smaller pieces (chunks) for processing, then combining them back together. Necessary because models have limited context windows.",
            "analogy": "Like reading a long book chapter by chapter, then putting the chapters together to understand the whole story."
        }
    ]
    
    for i, term_info in enumerate(terms):
        st.markdown(f"""
        <div class="term-definition">
            <h3 style="color: #667eea; margin-bottom: 0.5rem;">{term_info['term']}</h3>
            <p><strong>Definition:</strong> {term_info['definition']}</p>
            <p style="font-style: italic; color: #666;"><strong>Analogy:</strong> {term_info['analogy']}</p>
        </div>
        """, unsafe_allow_html=True)

def show_conversion_demo():
    """Interactive conversion demo"""
    st.markdown('<h2 class="section-header">🎵 Try Voice Conversion Yourself</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <p>Upload your audio files to see the voice conversion process in action. You can use the example files 
        or upload your own!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Source Audio")
        st.markdown("**Voice to convert** (the voice you want to change)")
        source_file = st.file_uploader(
            "Upload source audio", 
            type=['wav', 'mp3', 'flac', 'm4a'],
            key="source_upload"
        )
        
        # Example files
        if st.button("📁 Use Example Source"):
            example_source = "examples/source/source_s1.wav"
            if os.path.exists(example_source):
                with open(example_source, "rb") as f:
                    st.download_button("Download Example", f.read(), "example_source.wav", "audio/wav")
    
    with col2:
        st.subheader("🎯 Reference Audio")
        st.markdown("**Target voice** (the voice you want to sound like, 1-30 seconds)")
        reference_file = st.file_uploader(
            "Upload reference audio", 
            type=['wav', 'mp3', 'flac', 'm4a'],
            key="reference_upload"
        )
        
        # Example files
        if st.button("📁 Use Example Reference"):
            example_ref = "examples/reference/s1p1.wav"
            if os.path.exists(example_ref):
                with open(example_ref, "rb") as f:
                    st.download_button("Download Example", f.read(), "example_reference.wav", "audio/wav")
    
    # Parameters
    st.markdown("---")
    st.subheader("⚙️ Conversion Parameters")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        diffusion_steps = st.slider("Diffusion Steps", 10, 50, 30, 
                                    help="More steps = better quality but slower. 25-30 recommended.")
        length_adjust = st.slider("Length Adjust", 0.8, 1.2, 1.0, 0.05,
                                  help="Speed adjustment: <1.0 = faster, >1.0 = slower")
    with col2:
        inference_cfg_rate = st.slider("CFG Rate", 0.0, 1.0, 0.7, 0.1,
                                     help="Controls voice similarity vs content preservation")
        f0_condition = st.checkbox("F0 Conditioning", False,
                                  help="Enable for singing voice conversion")
    with col3:
        auto_f0_adjust = st.checkbox("Auto F0 Adjust", False,
                                    help="Automatically adjust pitch to match reference")
        preprocess = st.checkbox("Preprocess Audio", True,
                                help="Apply noise reduction and normalization")
    
    # Process button
    if st.button("🚀 Start Conversion", type="primary", use_container_width=True):
        if source_file and reference_file:
            with st.spinner("Loading audio files..."):
                source_audio, source_sr = load_audio_file(source_file)
                ref_audio, ref_sr = load_audio_file(reference_file)
                
                if source_audio is not None and ref_audio is not None:
                    st.session_state.source_audio = (source_audio, source_sr)
                    st.session_state.reference_audio = (ref_audio, ref_sr)
                    
                    # Show audio info
                    st.success("✅ Audio files loaded successfully!")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.audio(source_file, format='audio/wav')
                        st.caption(f"Source: {len(source_audio)/source_sr:.2f}s, {source_sr}Hz")
                    with col2:
                        st.audio(reference_file, format='audio/wav')
                        st.caption(f"Reference: {len(ref_audio)/ref_sr:.2f}s, {ref_sr}Hz")
                    
                    # Perform conversion
                    perform_conversion(
                        source_audio, source_sr,
                        ref_audio, ref_sr,
                        diffusion_steps, length_adjust, 
                        inference_cfg_rate, f0_condition,
                        auto_f0_adjust, preprocess
                    )
        else:
            st.error("⚠️ Please upload both source and reference audio files!")
    
    # Show visualization if audio is loaded
    if st.session_state.source_audio is not None:
        st.markdown("---")
        st.subheader("📊 Audio Visualization")
        
        source_audio, source_sr = st.session_state.source_audio
        ref_audio, ref_sr = st.session_state.reference_audio
        
        tab1, tab2, tab3 = st.tabs(["Waveforms", "Spectrograms", "F0 Contours"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(plot_waveform(source_audio, source_sr, "Source Audio Waveform"))
            with col2:
                st.pyplot(plot_waveform(ref_audio, ref_sr, "Reference Audio Waveform"))
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(plot_spectrogram(source_audio, source_sr, "Source Mel Spectrogram"))
            with col2:
                st.pyplot(plot_spectrogram(ref_audio, ref_sr, "Reference Mel Spectrogram"))
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(plot_f0_contour(source_audio, source_sr, "Source F0 Contour"))
            with col2:
                st.pyplot(plot_f0_contour(ref_audio, ref_sr, "Reference F0 Contour"))

def perform_conversion(source_audio, source_sr, ref_audio, ref_sr, 
                      diffusion_steps, length_adjust, inference_cfg_rate,
                      f0_condition, auto_f0_adjust, preprocess):
    """Perform the actual voice conversion"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Save temporary files
        temp_source = "temp_source.wav"
        temp_ref = "temp_ref.wav"
        output_dir = "audio_outputs/results"
        os.makedirs(output_dir, exist_ok=True)
        
        sf.write(temp_source, source_audio, source_sr)
        sf.write(temp_ref, ref_audio, ref_sr)
        
        status_text.text("🔄 Step 1/9: Preprocessing audio...")
        progress_bar.progress(10)
        
        # Import conversion function
        from scripts.convert_voice import convert_voice
        
        status_text.text("🔄 Step 2/9: Extracting semantic content...")
        progress_bar.progress(20)
        
        status_text.text("🔄 Step 3/9: Extracting speaker embeddings...")
        progress_bar.progress(30)
        
        status_text.text("🔄 Step 4/9: Computing mel spectrograms...")
        progress_bar.progress(40)
        
        status_text.text("🔄 Step 5/9: Running diffusion model...")
        progress_bar.progress(60)
        
        # Perform conversion
        convert_voice(
            source_audio=temp_source,
            reference_audio=temp_ref,
            output_dir=output_dir,
            diffusion_steps=diffusion_steps,
            length_adjust=length_adjust,
            inference_cfg_rate=inference_cfg_rate,
            f0_condition=f0_condition,
            auto_f0_adjust=auto_f0_adjust,
            preprocess=preprocess
        )
        
        status_text.text("🔄 Step 6/9: Generating waveform with vocoder...")
        progress_bar.progress(80)
        
        # Find output file
        output_files = list(Path(output_dir).glob("*.wav"))
        if output_files:
            latest_output = max(output_files, key=os.path.getctime)
            
            status_text.text("✅ Conversion complete!")
            progress_bar.progress(100)
            
            # Load converted audio
            converted_audio, converted_sr = librosa.load(str(latest_output), sr=None)
            st.session_state.converted_audio = (converted_audio, converted_sr)
            st.session_state.conversion_complete = True
            
            st.success(f"🎉 Voice conversion completed! Output saved to: {latest_output}")
            
            # Play audio
            st.audio(str(latest_output), format='audio/wav')
            
            # Cleanup
            if os.path.exists(temp_source):
                os.remove(temp_source)
            if os.path.exists(temp_ref):
                os.remove(temp_ref)
        else:
            st.error("Conversion completed but output file not found.")
            
    except Exception as e:
        st.error(f"❌ Error during conversion: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def show_results():
    """Show conversion results and comparisons"""
    st.markdown('<h2 class="section-header">📊 Conversion Results</h2>', unsafe_allow_html=True)
    
    if not st.session_state.conversion_complete:
        st.info("👆 Please perform a conversion first in the 'Try It Yourself' section!")
        return
    
    st.success("✅ Conversion results available!")
    
    # Audio comparison
    st.subheader("🎵 Audio Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📤 Source Audio")
        if st.session_state.source_audio:
            audio, sr = st.session_state.source_audio
            st.audio(audio, sample_rate=sr, format='audio/wav')
            st.caption(f"Duration: {len(audio)/sr:.2f}s")
    
    with col2:
        st.markdown("### 🎯 Reference Audio")
        if st.session_state.reference_audio:
            audio, sr = st.session_state.reference_audio
            st.audio(audio, sample_rate=sr, format='audio/wav')
            st.caption(f"Duration: {len(audio)/sr:.2f}s")
    
    with col3:
        st.markdown("### ✨ Converted Audio")
        if st.session_state.converted_audio:
            audio, sr = st.session_state.converted_audio
            st.audio(audio, sample_rate=sr, format='audio/wav')
            st.caption(f"Duration: {len(audio)/sr:.2f}s")
    
    # Visual comparison
    st.subheader("📈 Visual Comparison")
    
    if st.session_state.source_audio and st.session_state.converted_audio:
        source_audio, source_sr = st.session_state.source_audio
        converted_audio, converted_sr = st.session_state.converted_audio
        
        tab1, tab2, tab3 = st.tabs(["Waveforms", "Spectrograms", "F0 Comparison"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(plot_waveform(source_audio, source_sr, "Source Waveform"))
            with col2:
                st.pyplot(plot_waveform(converted_audio, converted_sr, "Converted Waveform"))
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(plot_spectrogram(source_audio, source_sr, "Source Spectrogram"))
            with col2:
                st.pyplot(plot_spectrogram(converted_audio, converted_sr, "Converted Spectrogram"))
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(plot_f0_contour(source_audio, source_sr, "Source F0"))
            with col2:
                st.pyplot(plot_f0_contour(converted_audio, converted_sr, "Converted F0"))
    
    # Analysis
    st.subheader("🔍 Analysis")
    st.markdown("""
    <div class="info-box">
        <h3>What to Listen For:</h3>
        <ul>
            <li><strong>Voice Similarity:</strong> Does the converted voice sound like the reference speaker?</li>
            <li><strong>Content Preservation:</strong> Are the words and meaning preserved correctly?</li>
            <li><strong>Naturalness:</strong> Does the output sound natural, without artifacts or robotic quality?</li>
            <li><strong>Pitch & Prosody:</strong> Is the rhythm and intonation appropriate?</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

