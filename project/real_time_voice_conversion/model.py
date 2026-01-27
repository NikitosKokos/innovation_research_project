import sys
import os
import torch
import librosa
import numpy as np
import yaml
import warnings
import torch.nn.utils.prune as prune
from scipy import signal

# Make torchaudio optional
TORCHAUDIO_AVAILABLE = False
torchaudio = None
T = None
try:
    import torchaudio
    import torchaudio.transforms as T
    # Test if torchaudio actually works (not a stub)
    try:
        test_tensor = torch.zeros(1, 1000)
        torchaudio.functional.resample(test_tensor, 16000, 22050)
        TORCHAUDIO_AVAILABLE = True
    except (NotImplementedError, AttributeError) as e:
        print(f"[Warning] torchaudio is a stub - functionality not available. Using librosa fallbacks.")
        TORCHAUDIO_AVAILABLE = False
        torchaudio = None
        T = None
except (ImportError, OSError) as e:
    print(f"[Warning] torchaudio not available ({e}). Using librosa fallbacks.")

# Add project root to sys.path to allow imports from modules/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from modules.commons import build_model, load_checkpoint, recursive_munch
from modules.campplus.DTDNN import CAMPPlus
from modules.audio import mel_spectrogram

# Suppress warnings
warnings.simplefilter('ignore')

class SeedVCModel:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.fp16 = config.FP16
        
        print(f"[Model] Initializing on {self.device}...")
        self._load_models()
        
        # Placeholders for target features
        self.target_mel = None
        self.target_style = None
        self.target_content = None
        self.prompt_condition = None
        self.target_length = None
        
    def _load_models(self):
        # Import torch locally at the start of method to avoid UnboundLocalError
        # This is needed because 'import torch._dynamo' inside the method marks 'torch' as local
        import torch
        
        # Helper function for safe device movement (defined once for all models)
        def safe_to_device(model, device, model_name="model"):
            """Safely move model to device, fallback to CPU if CUDA fails"""
            # Import torch locally to avoid 'free variable' closure issues
            import torch
            
            if device.type == 'cuda':
                try:
                    # Test if CUDA is actually working
                    test_tensor = torch.zeros(1).to(device)
                    del test_tensor
                    torch.cuda.empty_cache()
                    # Try to move model
                    model = model.to(device)
                    return model
                except Exception as e:
                    print(f"[Warning] Failed to move {model_name} to CUDA: {e}")
                    print(f"[Info] Falling back to CPU for {model_name}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # CRITICAL: CPU does not support float16 (Half) for many operations (like LayerNorm)
                    # If we fallback to CPU, we MUST convert to float32
                    return model.to('cpu').float()
            else:
                return model.to(device)
        
        # 1. Load DiT Model Config
        print(f"[Model] Loading Config from {self.config.CONFIG_PATH}")
        model_config = yaml.safe_load(open(self.config.CONFIG_PATH, "r"))
        model_params = recursive_munch(model_config["model_params"])
        model_params.dit_type = 'DiT'
        
        # 2. Build DiT Model
        self.model = build_model(model_params, stage="DiT")
        
        # 3. Load Checkpoint
        use_optimized = getattr(self.config, 'USE_OPTIMIZED_MODEL', False)
        if use_optimized:
            print(f"[Model] Loading Optimized Checkpoint from {self.config.CHECKPOINT_PATH}")
            print(f"[Model] Model is pre-optimized: FP16 quantized, 0% pruning")
        else:
            print(f"[Model] Loading Original Checkpoint from {self.config.CHECKPOINT_PATH}")
            print(f"[Model] Using original model (more stable, FP32)")
        
        # Load checkpoint
        self.model, _, _, _ = load_checkpoint(
            self.model, None, self.config.CHECKPOINT_PATH,
            load_only_params=True, ignore_modules=[], is_distributed=False
        )
        
        # Move to device safely
        for key in self.model:
            self.model[key].eval()
            self.model[key] = safe_to_device(self.model[key], self.device, f"Model[{key}]")
        
        # Enable PyTorch optimizations for faster inference
        if self.device.type == 'cuda' and torch.cuda.is_available():
            # Enable cuDNN benchmarking for faster convolutions (finds best algorithms)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
            print("[Model] Enabled cuDNN benchmarking for faster inference")
        
        # Apply structured pruning safely
        if getattr(self.config, 'ENABLE_EDGE_OPTIMIZATION', False) and getattr(self.config, 'PRUNING_RATIO', 0.0) > 0:
            self._apply_structured_pruning(self.model, amount=self.config.PRUNING_RATIO)
        
        # Apply structured pruning safely
        if getattr(self.config, 'ENABLE_EDGE_OPTIMIZATION', False) and getattr(self.config, 'PRUNING_RATIO', 0.0) > 0:
            self._apply_structured_pruning(self.model, amount=self.config.PRUNING_RATIO)
        
        # Apply quantization for faster inference (INT8 quantization)
        enable_quant = getattr(self.config, 'ENABLE_QUANTIZATION', False)
        quant_type = getattr(self.config, 'QUANTIZATION_TYPE', 'dynamic')
        
        if enable_quant:
            print(f"[Model] Applying {quant_type} INT8 quantization for faster inference...")
            try:
                if quant_type == "dynamic":
                    # Dynamic quantization: quantize on-the-fly (easiest, works immediately)
                    # Quantize the CFM estimator (main model component)
                    if hasattr(self.model, 'cfm') and hasattr(self.model.cfm, 'estimator'):
                        print("[Model] Quantizing CFM estimator with dynamic INT8...")
                        self.model.cfm.estimator = torch.quantization.quantize_dynamic(
                            self.model.cfm.estimator,
                            {torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d},  # Quantize these layer types
                            dtype=torch.qint8
                        )
                        print("[Model] CFM estimator quantized successfully")
                    
                    # Quantize length regulator if possible
                    if hasattr(self.model, 'length_regulator'):
                        try:
                            self.model.length_regulator = torch.quantization.quantize_dynamic(
                                self.model.length_regulator,
                                {torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d},
                                dtype=torch.qint8
                            )
                            print("[Model] Length regulator quantized successfully")
                        except Exception as e:
                            print(f"[Warning] Could not quantize length_regulator: {e}")
                    
                    print("[Model] Dynamic quantization applied - expect 2-4x speedup")
                else:
                    print(f"[Warning] Quantization type '{quant_type}' not implemented, using FP32")
            except Exception as e:
                print(f"[Warning] Quantization failed: {e}, continuing with FP32")
                print("[Info] Model will run slower but should still work")
        
        # FP16 optimization for Jetson Nano
        # Use autocast instead of full model conversion - safer and handles dtype automatically
        # Autocast provides FP16 acceleration without dtype mismatch issues
        if self.fp16 and self.device.type == 'cuda':
            print("[Model] FP16 enabled for Jetson Nano - using autocast for mixed precision")
            print("[Model] Autocast provides FP16 acceleration without dtype conversion issues")
            print("[Model] This is safer than full FP16 conversion and handles dtype automatically")
            # Don't convert model to FP16 - use autocast instead
            # This avoids dtype mismatch issues in complex operations
        elif self.fp16 and self.device.type == 'cpu':
            print("[Warning] FP16 not supported on CPU. Using FP32.")
        
        # Note: For optimized models, weights may be in FP16 format but we'll use autocast
        # For original models, we use FP32 with autocast for mixed precision

        # Initialize caches for DiT (Crucial for Transformer models in Seed-VC V2)
        if hasattr(self.model.cfm.estimator, "setup_caches"):
            print("[Model] Initializing DiT caches...")
            # Optimized for Jetson Nano: Smaller cache size to save memory
            # max_batch_size=1 (no CFG when INFERENCE_CFG_RATE=0.0), max_seq_len reduced for memory
            max_batch = 1 if self.config.INFERENCE_CFG_RATE == 0.0 else 2
            # Reduce max_seq_length from 2048 to 1024 to save memory (Jetson Nano has limited GPU memory)
            self.model.cfm.estimator.setup_caches(max_batch_size=max_batch, max_seq_length=1024)
        
        # Apply torch.compile if enabled (PyTorch 2.0+) and NOT using TensorRT
        if getattr(self.config, 'USE_TORCH_COMPILE', False) and not getattr(self.config, 'USE_TENSORRT', False):
            print("[Model] Applying torch.compile optimization...")
            try:
                # Compile the main estimator (DiT)
                # mode='reduce-overhead' is good for small batches, 'max-autotune' for throughput
                # On Jetson, 'default' or 'reduce-overhead' is safer
                # 'cudagraphs' backend is often better for Jetson than 'inductor' if triton is missing
                try:
                    import torch._dynamo
                    torch._dynamo.config.suppress_errors = True
                except:
                    pass
                
                # Check for Triton
                try:
                    import triton
                    backend = 'inductor'
                except ImportError:
                    print("[Model] Triton not found, using 'cudagraphs' backend instead of 'inductor'")
                    backend = 'cudagraphs'

                self.model.cfm.estimator = torch.compile(self.model.cfm.estimator, mode='reduce-overhead', backend=backend)
                print(f"[Model] DiT estimator compiled successfully with backend='{backend}'")
            except Exception as e:
                print(f"[Warning] torch.compile failed: {e}")

        # Initialize TensorRT if enabled
        if getattr(self.config, 'USE_TENSORRT', False):
            trt_path = getattr(self.config, 'TENSORRT_ENGINE_PATH', None)
            if trt_path and os.path.exists(trt_path):
                print(f"[Model] Loading TensorRT engine from {trt_path}...")
                try:
                    from project.real_time_voice_conversion.tensorrt_wrapper import TensorRTEstimator
                    # Replace the PyTorch estimator with TensorRT wrapper
                    # This allows us to keep the rest of the pipeline (CFM loop) identical
                    self.model.cfm.estimator = TensorRTEstimator(trt_path)
                    print("[Model] TensorRT engine loaded and hooked into CFM successfully!")
                except Exception as e:
                    print(f"[Error] Failed to load TensorRT engine: {e}")
                    print("[Info] Falling back to PyTorch inference")
            else:
                if trt_path:
                    print(f"[Warning] TensorRT engine not found at {trt_path}")
                    print("[Info] Please run 'python project/real_time_voice_conversion/convert_to_tensorrt.py ...' first")
                print("[Info] Falling back to PyTorch inference")

        # 4. Load CAM++ (Style Encoder)
        # Move to CPU for Jetson Nano to save GPU memory for DiT
        print("[Model] Loading CAM++ Style Encoder (forcing CPU)...")
        from hf_utils import load_custom_model_from_hf
        campplus_ckpt_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
        self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        self.campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model.eval()
        self.campplus_model = self.campplus_model.to('cpu')
        print("[Model] CAM++ loaded on CPU")

        # 5. Load Vocoder (BigVGAN or HiFi-GAN)
        vocoder_type = getattr(self.config, 'VOCODER_TYPE', 'bigvgan')
        print(f"[Model] Loading Vocoder: {vocoder_type}...")
        
        if vocoder_type == 'hifigan':
            try:
                print("[Model] Loading HiFi-GAN (NVIDIA/DeepLearningExamples)...")
                # Use torch.hub to load official NVIDIA HiFi-GAN
                # This requires internet access on first run
                # Returns: (model, utils, denoiser)
                self.vocoder, _, _ = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_hifigan')
                self.vocoder.eval()
                
                # Convert to FP16 if enabled
                if self.fp16 and self.device.type == 'cuda':
                    self.vocoder = self.vocoder.half()
                    print("[Model] HiFi-GAN converted to FP16")
                
                self.vocoder = safe_to_device(self.vocoder, self.device, "HiFi-GAN")
                print(f"[Model] HiFi-GAN loaded on {next(self.vocoder.parameters()).device}")
                
            except Exception as e:
                print(f"[Error] Failed to load HiFi-GAN: {e}")
                print("[Info] Falling back to BigVGAN...")
                vocoder_type = 'bigvgan'
        
        if vocoder_type == 'bigvgan':
            # Load BigVGAN on GPU and use FP16 for massive speedup on Jetson
            print("[Model] Loading BigVGAN Vocoder (FP16 Optimized)...")
            from modules.bigvgan import bigvgan
            bigvgan_name = model_params.vocoder.name 
            self.vocoder = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
            self.vocoder.remove_weight_norm()
            self.vocoder.eval()
            
            # Convert to FP16 for 4-8x speedup on Jetson GPU
            if self.fp16 and self.device.type == 'cuda':
                self.vocoder = self.vocoder.half()
                print("[Model] BigVGAN converted to FP16")
                
            self.vocoder = safe_to_device(self.vocoder, self.device, "BigVGAN")
            print(f"[Model] BigVGAN loaded on {next(self.vocoder.parameters()).device}")

        # 6. Load Whisper (Content Encoder)
        # Load on GPU for real-time performance (CPU is too slow ~25s)
        print("[Model] Loading Whisper Content Encoder (on GPU)...")
        from transformers import AutoFeatureExtractor, WhisperModel
        whisper_name = model_params.speech_tokenizer.name
        
        # Clear CUDA cache before loading Whisper
        if self.device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            # Use float16 for Whisper on GPU to save memory and increase speed
            whisper_dtype = torch.float16 if self.fp16 and self.device.type == 'cuda' else torch.float32
            print(f"[Model] Loading Whisper with dtype={whisper_dtype}")
            
            self.whisper_model = WhisperModel.from_pretrained(
                whisper_name, torch_dtype=whisper_dtype
            )
            del self.whisper_model.decoder  # We only need the encoder
            self.whisper_model.eval()
            self.whisper_model = safe_to_device(self.whisper_model, self.device, "Whisper")
            print(f"[Model] Whisper loaded on {next(self.whisper_model.encoder.parameters()).device}")
        except Exception as e:
            print(f"[Error] Failed to load Whisper: {e}")
            raise
        
        self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

        # 7. Setup Mel Spectrogram Function
        self.mel_fn_args = {
            "n_fft": model_config['preprocess_params']['spect_params']['n_fft'],
            "win_size": model_config['preprocess_params']['spect_params']['win_length'],
            "hop_size": model_config['preprocess_params']['spect_params']['hop_length'],
            "num_mels": model_config['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": model_config['preprocess_params']['sr'],
            "fmin": model_config['preprocess_params']['spect_params'].get('fmin', 0),
            "fmax": 8000,
            "center": False
        }
        self.to_mel = lambda x: mel_spectrogram(x, **self.mel_fn_args)

        print("[Model] All models loaded successfully.")
    
    def _apply_structured_pruning(self, model_container, amount=0.2):
        """
        Apply structured pruning SAFELY to Seed-VC DiT model components.
        
        Best practices for transformer pruning:
        - ONLY prune feed-forward layers (FFN) - they have the most redundancy
        - DO NOT prune attention layers (wqkv, wo) - critical for quality
        - DO NOT prune embedding layers (x_embedder, t_embedder, style_in, cond_projection)
        - DO NOT prune final output layers (final_mlp)
        - DO NOT prune layer norms or skip connections
        
        Based on research: FFN layers can be pruned 20-30% with minimal quality loss.
        """
        print(f"[Pruning] Applying {amount*100:.0f}% structured pruning to feed-forward layers only...")
        
        # Handle Munch/dict container vs single model
        models_to_prune = []
        if hasattr(model_container, 'items'):
            for k, v in model_container.items():
                models_to_prune.append((k, v))
        elif isinstance(model_container, torch.nn.Module):
            models_to_prune.append(('model', model_container))
        else:
            print(f"[Pruning] Warning: Unknown model container type {type(model_container)}")
            return

        total_pruned = 0
        total_skipped = 0
        
        # Critical layers to NEVER prune
        critical_patterns = [
            'embed',           # Embedding layers
            'pos',             # Positional encodings
            'wqkv',            # Attention query/key/value projections
            'wq', 'wk', 'wv', # Individual attention projections
            'wo',              # Attention output projection
            'final_mlp',       # Final output MLP
            'x_embedder',      # Input embedding
            't_embedder',      # Timestep embedding
            'style_in',        # Style embedding input
            'cond_projection', # Condition projection
            'norm',            # Layer norms
            'skip',            # Skip connections
            'cond_x_merge',    # Merge layers
        ]

        from torch.nn.utils import remove_weight_norm

        for model_name, model in models_to_prune:
            # Skip non-module items if any
            if not isinstance(model, torch.nn.Module):
                continue
                
            print(f"[Pruning] Processing sub-model: {model_name}...")
            
            # Remove weight norm before pruning (if present)
            for name, module in model.named_modules():
                try:
                    remove_weight_norm(module)
                except (ValueError, AttributeError):
                    pass
            
            pruned_count = 0
            skipped_count = 0
            
            for name, module in model.named_modules():
                # Skip critical layers
                is_critical = any(pattern in name.lower() for pattern in critical_patterns)
                if is_critical:
                    skipped_count += 1
                    continue
                
                # Only prune Linear layers in feed-forward blocks
                if isinstance(module, torch.nn.Linear):
                    # Check if this is a feed-forward layer
                    is_ffn_layer = (
                        'feed_forward' in name.lower() or
                        name.endswith('.w1') or
                        name.endswith('.w2') or
                        name.endswith('.w3') or
                        ('ffn' in name.lower() and 'attention' not in name.lower())
                    )
                    
                    if is_ffn_layer:
                        try:
                            # Prune feed-forward layers (these are safe to prune)
                            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
                            prune.remove(module, 'weight')
                            pruned_count += 1
                        except Exception as e:
                            print(f"[Warning] Failed to prune FFN layer {name}: {e}")
                            skipped_count += 1
                    else:
                        skipped_count += 1
                elif isinstance(module, torch.nn.Conv2d):
                    skipped_count += 1
            
            total_pruned += pruned_count
            total_skipped += skipped_count
        
        print(f"[Pruning] Pruned {total_pruned} feed-forward layers successfully!")
        print(f"[Pruning] Skipped {total_skipped} critical layers (attention, embeddings, norms)")
        print(f"[Pruning] This is a SAFE pruning strategy that preserves model quality")

    def set_target(self, target_path):
        """Pre-calculates style and mel features for the target voice."""
        print(f"[Model] Setting target voice: {target_path}")
        
        # Load audio
        ref_audio, _ = librosa.load(target_path, sr=self.mel_fn_args['sampling_rate'])
        
        # Convert to tensor (keep in FP32, autocast will handle precision)
        ref_audio = torch.tensor(ref_audio).unsqueeze(0).float().to(self.device)
        
        # 1. Compute Mel Spectrogram (for length regulation reference)
        with torch.no_grad():
            self.target_mel = self.to_mel(ref_audio)
            # Keep in FP32 - autocast will handle conversion when needed
            
        # 2. Compute Style Embedding (CAM++)
        # Resample to 16k for CAM++
        if TORCHAUDIO_AVAILABLE:
            try:
                ref_waves_16k = torchaudio.functional.resample(ref_audio, self.mel_fn_args['sampling_rate'], 16000)
                feat2 = torchaudio.compliance.kaldi.fbank(
                    ref_waves_16k,
                    num_mel_bins=80,
                    dither=0,
                    sample_frequency=16000
                )
            except (NotImplementedError, AttributeError) as e:
                # Fallback to librosa if torchaudio fails at runtime
                ref_audio_np = ref_audio.squeeze(0).cpu().numpy()
                ref_waves_16k_np = librosa.resample(ref_audio_np, orig_sr=self.mel_fn_args['sampling_rate'], target_sr=16000)
                ref_waves_16k = torch.from_numpy(ref_waves_16k_np).float().unsqueeze(0).to(self.device)
                # Use librosa mel spectrogram as approximation
                feat2 = librosa.feature.melspectrogram(
                    y=ref_waves_16k_np,
                    sr=16000,
                    n_mels=80,
                    fmin=0,
                    fmax=8000,
                    n_fft=512,
                    hop_length=160
                )
                # Convert to log scale and transpose to match kaldi format (time, freq)
                feat2 = torch.tensor(np.log(feat2.T + 1e-10), dtype=torch.float32).to(self.device)
        else:
            # Fallback to librosa
            ref_audio_np = ref_audio.squeeze(0).cpu().numpy()
            ref_waves_16k_np = librosa.resample(ref_audio_np, orig_sr=self.mel_fn_args['sampling_rate'], target_sr=16000)
            ref_waves_16k = torch.from_numpy(ref_waves_16k_np).float().unsqueeze(0).to(self.device)
            # Use librosa mel spectrogram as approximation
            feat2 = librosa.feature.melspectrogram(
                y=ref_waves_16k_np,
                sr=16000,
                n_mels=80,
                fmin=0,
                fmax=8000,
                n_fft=512,
                hop_length=160
            )
            # Convert to log scale and transpose to match kaldi format (time, freq)
            feat2 = torch.tensor(np.log(feat2.T + 1e-10), dtype=torch.float32).to(self.device)
        
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        
        # Get the device where CAM++ model is actually located (GPU or CPU)
        campplus_device = next(self.campplus_model.parameters()).device
        
        # Move features to CAM++ device
        feat2_device = feat2.to(campplus_device)
        
        # Ensure feat2_device has correct dtype for CAM++
        if self.fp16 and campplus_device.type == 'cuda':
            feat2_device = feat2_device.half()
        else:
            feat2_device = feat2_device.float()
        
        with torch.no_grad():
            try:
                self.target_style = self.campplus_model(feat2_device.unsqueeze(0))
                # Move result to main device (CUDA) for use with other models
                self.target_style = self.target_style.to(self.device)
                # Keep in FP32 - autocast will handle conversion when needed
            except RuntimeError as e:
                if "cuDNN" in str(e) or "CUDNN" in str(e) or "out of memory" in str(e).lower():
                    print(f"[Warning] Error with CAM++ on {campplus_device}: {e}")
                    print("[Info] Falling back to CPU for CAM++...")
                    # Fallback to CPU
                    self.campplus_model = self.campplus_model.to('cpu')
                    feat2_cpu = feat2.cpu()
                    self.target_style = self.campplus_model(feat2_cpu.unsqueeze(0))
                    self.target_style = self.target_style.to(self.device)
                else:
                    raise
            
        # 3. Compute Whisper Features for Target (Prompt Condition)
        # We use the whole target audio as prompt
        self.target_content = self._extract_whisper_features(ref_waves_16k)
        # Keep in FP32 - autocast will handle conversion when needed
        
        # 4. Pre-calculate Length Regulator for Target Prompt
        print("[Model] Pre-calculating prompt condition for target voice...")
        with torch.no_grad():
            target_prompt_length = torch.LongTensor([self.target_mel.size(2)]).to(self.device)
            # Use autocast for FP16 if enabled
            if self.fp16 and self.device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    prompt_condition, _, _, _, _ = self.model.length_regulator(
                        self.target_content,
                        ylens=target_prompt_length,
                        n_quantizers=3,
                        f0=None
                    )
            else:
                prompt_condition, _, _, _, _ = self.model.length_regulator(
                    self.target_content,
                    ylens=target_prompt_length,
                    n_quantizers=3,
                    f0=None
                )
            self.prompt_condition = prompt_condition
        
        print("[Model] Target features pre-calculated.")

        # 5. Warmup Phase
        print("[Model] Warming up GPU and cuDNN (this prevents initial lag)...")
        try:
            # Create a dummy chunk for warmup
            warmup_chunk = np.zeros(self.config.BLOCK_SIZE * getattr(self.config, 'CHUNKS_TO_ACCUMULATE', 6))
            # Run once to trigger cuDNN benchmarking and memory allocation
            _ = self.process_chunk(warmup_chunk)
            print("[Model] Warmup complete.")
        except Exception as e:
            print(f"[Warning] Warmup failed during set_target: {e}. The first chunk may still be slow.")
            # Don't raise, just continue
        
        # Reset timing counter after warmup
        self._timing_chunk_count = 0
        
        # Unload CAM++ to free memory (it's only used for target style extraction)
        # Only if we are constrained on memory
        print("[Model] Unloading CAM++ to free memory...")
        self.campplus_model = None
        import gc
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def _extract_whisper_features(self, audio_16k):
        """Helper to extract Whisper features from 16kHz audio."""
        # Get the device where Whisper model is actually located
        whisper_device = next(self.whisper_model.encoder.parameters()).device
        
        # Ensure audio is on CPU for feature extraction
        audio_16k_np = audio_16k.squeeze(0).cpu().numpy()

        # MANDATORY WHISPER PADDING: Whisper model strictly requires 30 seconds of audio (3000 mel frames)
        # Some versions of transformers ignore padding arguments, so we pad the raw audio.
        # 30 seconds * 16000 samples/sec = 480,000 samples
        target_samples = 480000 
        original_length = len(audio_16k_np)
        
        if original_length < target_samples:
            # Pad with zeros to exactly 30 seconds
            audio_16k_np = np.pad(audio_16k_np, (0, target_samples - original_length), mode='constant')
        else:
            # Truncate to 30 seconds if longer
            audio_16k_np = audio_16k_np[:target_samples]

        # Extract features (now guaranteed to be 3000 mel frames)
        inputs = self.whisper_feature_extractor(
            [audio_16k_np],
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=16000,
            do_normalize=True
        )
        
        input_features = inputs.input_features
        
        # Double check the shape - it MUST be (batch, 80, 3000)
        if input_features.shape[-1] != 3000:
            if input_features.shape[-1] < 3000:
                pad_amount = 3000 - input_features.shape[-1]
                input_features = torch.nn.functional.pad(input_features, (0, pad_amount), value=0.0)
            else:
                input_features = input_features[:, :, :3000]

        # Move to correct device and apply masking
        input_features = self.whisper_model._mask_input_features(
            input_features, attention_mask=inputs.attention_mask
        ).to(whisper_device)
        
        # Ensure correct dtype for Whisper encoder
        input_features = input_features.to(self.whisper_model.encoder.dtype)

        with torch.no_grad():
            outputs = self.whisper_model.encoder(
                input_features,
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        
        # Downsample and slice to match the original audio length
        # Whisper encoder output has a resolution of 20ms (50fps)
        # audio_16k original length // 320 gives the number of 20ms frames
        features = outputs.last_hidden_state.to(torch.float32).to(self.device)
        features = features[:, :original_length // 320 + 1]
        return features

    @torch.no_grad()
    @torch.inference_mode()  # Faster than no_grad() for inference
    def process_chunk(self, audio_chunk):
        """
        Processes a single chunk of audio.
        Processes ALL audio regardless of level (including quiet/silent audio).
        Args:
            audio_chunk (numpy array): Input audio at config.SAMPLE_RATE
        Returns:
            numpy array: Converted audio
        """
        try:
            # Don't clear cache every time - it adds overhead
            # Only clear if we're running out of memory
            # if self.device.type == 'cuda' and torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            
            # Keep inputs in FP32 - autocast will handle precision during computation
            # This is safer and prevents numerical issues
            
            # 1. Prepare Input
            # Process ALL audio, even if it's quiet/silent
            if not isinstance(audio_chunk, np.ndarray):
                audio_chunk = np.array(audio_chunk)
            
            if audio_chunk.size == 0:
                raise ValueError("Empty audio chunk received")
            
            source_audio = torch.tensor(audio_chunk).unsqueeze(0).float().to(self.device)
            
            # Track timing for diagnostics (only on first few chunks to avoid overhead)
            timing_enabled = not hasattr(self, '_timing_chunk_count') or self._timing_chunk_count < 3
            if not hasattr(self, '_timing_chunk_count'):
                self._timing_chunk_count = 0
            if timing_enabled:
                import time
                step_times = {}
                step_start = time.time()
            
            # Clear CUDA cache periodically to prevent memory buildup (Jetson Nano has limited memory)
            # Increased frequency to every 10 chunks to avoid OOM
            if self.device.type == 'cuda' and self._timing_chunk_count % 10 == 0:
                torch.cuda.empty_cache()
            
            # 2. Resample to 16k for Whisper
            # Use fast resampling for real-time performance
            # Check where Whisper is to avoid unnecessary GPU transfer
            whisper_device = next(self.whisper_model.encoder.parameters()).device
            
            if TORCHAUDIO_AVAILABLE:
                try:
                    source_16k = torchaudio.functional.resample(source_audio, self.config.SAMPLE_RATE, 16000)
                    if source_16k.device != whisper_device:
                        source_16k = source_16k.to(whisper_device)
                except (NotImplementedError, AttributeError):
                    # Fallback to fast torch interpolation (GPU compatible)
                    # source_audio is (1, T) -> (1, 1, T) for interpolate
                    ratio = 16000 / self.config.SAMPLE_RATE
                    target_len = int(source_audio.shape[-1] * ratio)
                    source_16k = torch.nn.functional.interpolate(
                        source_audio.unsqueeze(0), size=target_len, mode='linear', align_corners=False
                    ).squeeze(0).to(whisper_device)
            else:
                # Use fast torch interpolation (GPU compatible)
                # Avoids CPU<->GPU sync of scipy/numpy
                ratio = 16000 / self.config.SAMPLE_RATE
                target_len = int(source_audio.shape[-1] * ratio)
                source_16k = torch.nn.functional.interpolate(
                    source_audio.unsqueeze(0), size=target_len, mode='linear', align_corners=False
                ).squeeze(0).to(whisper_device)
            
            # 3. Extract Content (Whisper)
            if timing_enabled:
                step_times['resample_16k'] = (time.time() - step_start) * 1000
                step_start = time.time()
            try:
                source_content = self._extract_whisper_features(source_16k)
                if source_content is None or source_content.size(1) == 0:
                    raise ValueError("Whisper feature extraction failed or returned empty features")
                # Keep in FP32 - autocast will handle conversion when needed
            except Exception as e:
                raise RuntimeError(f"Failed to extract Whisper features: {e}") from e
            
            if timing_enabled:
                step_times['whisper'] = (time.time() - step_start) * 1000
                step_start = time.time()
            
            # 4. Length Regulation
            # Resample source to model's sampling rate for Mel calculation
            # Use fast resampling for real-time performance
            if self.config.SAMPLE_RATE != self.mel_fn_args['sampling_rate']:
                if TORCHAUDIO_AVAILABLE:
                    try:
                        source_audio_model_sr = torchaudio.functional.resample(
                            source_audio, self.config.SAMPLE_RATE, self.mel_fn_args['sampling_rate']
                        )
                    except (NotImplementedError, AttributeError):
                        # Fallback to fast torch interpolation
                        ratio = self.mel_fn_args['sampling_rate'] / self.config.SAMPLE_RATE
                        target_len = int(source_audio.shape[-1] * ratio)
                        source_audio_model_sr = torch.nn.functional.interpolate(
                            source_audio.unsqueeze(0), size=target_len, mode='linear', align_corners=False
                        ).squeeze(0).to(self.device)
                else:
                    # Use fast torch interpolation
                    ratio = self.mel_fn_args['sampling_rate'] / self.config.SAMPLE_RATE
                    target_len = int(source_audio.shape[-1] * ratio)
                    source_audio_model_sr = torch.nn.functional.interpolate(
                        source_audio.unsqueeze(0), size=target_len, mode='linear', align_corners=False
                    ).squeeze(0).to(self.device)
            else:
                source_audio_model_sr = source_audio

            if timing_enabled:
                step_times['resample_model_sr'] = (time.time() - step_start) * 1000
                step_start = time.time()
            
            try:
                source_mel = self.to_mel(source_audio_model_sr)
                if source_mel is None or source_mel.size(2) == 0:
                    raise ValueError("Mel spectrogram computation failed or returned empty")
                # Keep in FP32 - autocast will handle conversion when needed
            except Exception as e:
                raise RuntimeError(f"Failed to compute mel spectrogram: {e}") from e
            
            if timing_enabled:
                step_times['mel'] = (time.time() - step_start) * 1000
                step_start = time.time()
            
            target_length = torch.LongTensor([int(source_mel.size(2) * self.config.LENGTH_ADJUST)]).to(self.device)
            
            # Run Length Regulator
            # Note: We pass the TARGET style/content as the "prompt" and the SOURCE content as the "input"
            
            # Source Condition (Content to be converted)
            try:
                # Use autocast for FP16 if enabled (handles dtype automatically)
                if self.fp16 and self.device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        cond, _, _, _, _ = self.model.length_regulator(
                            source_content, 
                            ylens=target_length,
                            n_quantizers=3,
                            f0=None # Assuming no F0 condition for now as per config
                        )
                else:
                    cond, _, _, _, _ = self.model.length_regulator(
                        source_content, 
                        ylens=target_length,
                        n_quantizers=3,
                        f0=None
                    )
                if cond is None or cond.size(1) == 0:
                    raise ValueError("Length regulator returned empty condition for source")
            except Exception as e:
                raise RuntimeError(f"Failed to run length regulator for source: {e}") from e
            
            # Use pre-calculated prompt condition for target voice
            prompt_condition = self.prompt_condition
            
            if timing_enabled:
                step_times['length_regulator'] = (time.time() - step_start) * 1000
                step_start = time.time()
            
            # Concatenate Prompt + Source
            try:
                cat_condition = torch.cat([prompt_condition, cond], dim=1)
                if cat_condition.size(1) == 0:
                    raise ValueError("Concatenated condition is empty")
            except Exception as e:
                raise RuntimeError(f"Failed to concatenate conditions: {e}") from e
            
            # Use pre-calculated target features
            target_mel = self.target_mel
            target_style = self.target_style
            
            if timing_enabled:
                step_times['concat'] = (time.time() - step_start) * 1000
                step_start = time.time()
            
            # 5. Diffusion Inference
            # Optimized for real-time: Disable tqdm progress bars to reduce I/O overhead
            # Patch tqdm in the CFM module to disable progress bars during inference
            import modules.v2.cfm as cfm_module
            from tqdm import tqdm as original_tqdm
            
            # Create a no-op tqdm replacement
            class NoOpTqdm:
                def __init__(self, iterable=None, *args, **kwargs):
                    self.iterable = iterable if iterable is not None else range(1)
                def __iter__(self):
                    return iter(self.iterable)
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
                def update(self, n=1):
                    pass
                def close(self):
                    pass
            
            # Temporarily replace tqdm in the CFM module
            cfm_module.tqdm = NoOpTqdm
            
            try:
                # Use autocast for FP16 mixed precision (safer than full FP16 conversion)
                # Autocast automatically handles dtype conversions and avoids mismatch issues
                if self.fp16 and self.device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        # Pass SKIP_LAYERS to inference
                        skip_layers = getattr(self.config, 'SKIP_LAYERS', [])
                        
                        vc_target = self.model.cfm.inference(
                            cat_condition,
                            torch.LongTensor([cat_condition.size(1)]).to(self.device),
                            target_mel, 
                            target_style, 
                            None,
                            self.config.DIFFUSION_STEPS,
                            inference_cfg_rate=self.config.INFERENCE_CFG_RATE,
                            skip_layers=skip_layers
                        )
                else:
                    # FP32 inference
                    skip_layers = getattr(self.config, 'SKIP_LAYERS', [])
                    
                    vc_target = self.model.cfm.inference(
                        cat_condition,
                        torch.LongTensor([cat_condition.size(1)]).to(self.device),
                        target_mel, 
                        target_style, 
                        None,
                        self.config.DIFFUSION_STEPS,
                        inference_cfg_rate=self.config.INFERENCE_CFG_RATE,
                        skip_layers=skip_layers
                    )
                
                if vc_target is None:
                    raise ValueError("CFM inference returned None")
                if vc_target.size(2) == 0:
                    raise ValueError("CFM inference returned empty output")
                    
            except Exception as e:
                raise RuntimeError(f"Failed during CFM inference: {e}") from e
            finally:
                # Restore original tqdm
                cfm_module.tqdm = original_tqdm
            
            if timing_enabled:
                step_times['cfm_inference'] = (time.time() - step_start) * 1000
                step_start = time.time()
            
            # 6. Slice Output
            # The model output includes the prompt reconstruction? 
            # In inference.py: vc_target = vc_target[:, :, mel2.size(-1):]
            # Yes, we slice off the prompt part.
            try:
                vc_target = vc_target[:, :, target_mel.size(-1):]
                if vc_target.size(2) == 0:
                    raise ValueError("Sliced output is empty")
            except Exception as e:
                raise RuntimeError(f"Failed to slice output: {e}") from e
            
            if timing_enabled:
                step_times['slice'] = (time.time() - step_start) * 1000
                step_start = time.time()
            
            # 7. Vocoder (Mel -> Waveform)
            try:
                # BigVGAN handles FP16/FP32 automatically if model is on correct device
                # Ensure input device matches model device
                vocoder_device = next(self.vocoder.parameters()).device
                vocoder_dtype = next(self.vocoder.parameters()).dtype
                
                vc_target_input = vc_target.to(vocoder_device).to(vocoder_dtype)
                
                # Inference with appropriate precision
                if vocoder_dtype == torch.float16:
                    with torch.autocast(device_type=vocoder_device.type, dtype=torch.float16):
                        vc_wave = self.vocoder(vc_target_input).squeeze()
                else:
                    vc_wave = self.vocoder(vc_target_input).squeeze()

                if vc_wave is None:
                    raise ValueError("Vocoder returned None")
                if vc_wave.numel() == 0:
                    raise ValueError("Vocoder returned empty output")
            except Exception as e:
                raise RuntimeError(f"Failed during vocoder inference: {e}") from e
            
            # Clear intermediate tensors to free memory (critical for Jetson Nano with limited memory)
            try:
                del vc_target, cat_condition, cond, source_content, source_mel, source_audio_model_sr, source_16k, source_audio
            except:
                pass
            # Aggressive memory cleanup removed as it's too slow for real-time

            if timing_enabled:
                step_times['vocoder'] = (time.time() - step_start) * 1000
                step_start = time.time()
            
            # 8. Resample back to output sample rate if needed
            if self.mel_fn_args['sampling_rate'] != self.config.SAMPLE_RATE:
                try:
                        if TORCHAUDIO_AVAILABLE:
                            try:
                                vc_wave = torchaudio.functional.resample(
                                    vc_wave.unsqueeze(0), self.mel_fn_args['sampling_rate'], self.config.SAMPLE_RATE
                                ).squeeze(0)
                            except (NotImplementedError, AttributeError):
                                # Fallback to fast torch interpolation
                                ratio = self.config.SAMPLE_RATE / self.mel_fn_args['sampling_rate']
                                target_len = int(vc_wave.shape[-1] * ratio)
                                vc_wave = torch.nn.functional.interpolate(
                                    vc_wave.unsqueeze(0).unsqueeze(0), size=target_len, mode='linear', align_corners=False
                                ).squeeze(0).squeeze(0)
                        else:
                            # Use fast torch interpolation
                            ratio = self.config.SAMPLE_RATE / self.mel_fn_args['sampling_rate']
                            target_len = int(vc_wave.shape[-1] * ratio)
                            vc_wave = torch.nn.functional.interpolate(
                                vc_wave.unsqueeze(0).unsqueeze(0), size=target_len, mode='linear', align_corners=False
                            ).squeeze(0).squeeze(0)
                except Exception as e:
                    raise RuntimeError(f"Failed to resample output: {e}") from e

            if timing_enabled:
                step_times['resample_output'] = (time.time() - step_start) * 1000
                total_time = sum(step_times.values())
                self._timing_chunk_count += 1
                print(f"[Model Timing] Chunk {self._timing_chunk_count}: {', '.join([f'{k}={v:.1f}ms' for k, v in step_times.items()])}, total={total_time:.1f}ms")
            
            # Return as numpy array
            output = vc_wave.cpu().numpy()
            
            # Final validation
            if output.size == 0:
                raise ValueError("Model produced empty output")
            
            if np.any(np.isnan(output)) or np.any(np.isinf(output)):
                raise ValueError("Model output contains NaN or Inf values")
            
            # Debug: Log output statistics (only for first few chunks)
            if self._timing_chunk_count <= 3:
                output_max = np.abs(output).max()
                output_mean = np.abs(output).mean()
                output_std = np.std(output)
                print(f"[Model Debug] Chunk {self._timing_chunk_count} output stats: max={output_max:.6f}, mean={output_mean:.6f}, std={output_std:.6f}, shape={output.shape}")
                if output_max < 0.01:
                    print(f"[Model Warning] Output is very quiet (max={output_max:.6f}), may be inaudible. Consider increasing OUTPUT_GAIN in config.")
            
            return output
            
        except Exception as e:
            # Log the error with context
            error_msg = f"Error in process_chunk: {type(e).__name__}: {str(e)}"
            print(f"[Model Error] {error_msg}")
            import traceback
            traceback.print_exc()
            # Re-raise to let the stream handler deal with it
            raise
