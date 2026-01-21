import sys
import os
import torch
import torchaudio
import librosa
import numpy as np
import yaml
import warnings

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
        self.target_length = None
        
    def _load_models(self):
        # 1. Load DiT Model Config
        print(f"[Model] Loading Config from {self.config.CONFIG_PATH}")
        model_config = yaml.safe_load(open(self.config.CONFIG_PATH, "r"))
        model_params = recursive_munch(model_config["model_params"])
        model_params.dit_type = 'DiT'
        
        # 2. Build DiT Model
        self.model = build_model(model_params, stage="DiT")
        
        # 3. Load Checkpoint
        print(f"[Model] Loading Checkpoint from {self.config.CHECKPOINT_PATH}")
        self.model, _, _, _ = load_checkpoint(
            self.model, None, self.config.CHECKPOINT_PATH,
            load_only_params=True, ignore_modules=[], is_distributed=False
        )
        for key in self.model:
            self.model[key].eval()
            self.model[key].to(self.device)
            
        # 4. Load CAM++ (Style Encoder)
        print("[Model] Loading CAM++ Style Encoder...")
        # Assuming the standard path or downloading if needed. 
        # For this setup, we'll try to find the local file or assume it's cached/downloadable via the original logic
        # But to be safe and modular, we'll look for it in the project root if possible, or rely on hf_utils
        from hf_utils import load_custom_model_from_hf
        campplus_ckpt_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
        self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        self.campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model.eval()
        self.campplus_model.to(self.device)

        # 5. Load Vocoder (BigVGAN)
        print("[Model] Loading BigVGAN Vocoder...")
        from modules.bigvgan import bigvgan
        # Hardcoded for now based on typical usage, or could be in config
        bigvgan_name = model_params.vocoder.name 
        self.vocoder = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
        self.vocoder.remove_weight_norm()
        self.vocoder = self.vocoder.eval().to(self.device)

        # 6. Load Whisper (Content Encoder)
        print("[Model] Loading Whisper Content Encoder...")
        from transformers import AutoFeatureExtractor, WhisperModel
        whisper_name = model_params.speech_tokenizer.name
        self.whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(self.device)
        del self.whisper_model.decoder # We only need the encoder
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

    def set_target(self, target_path):
        """Pre-calculates style and mel features for the target voice."""
        print(f"[Model] Setting target voice: {target_path}")
        # Load audio
        ref_audio, _ = librosa.load(target_path, sr=self.mel_fn_args['sampling_rate'])
        
        # Convert to tensor
        ref_audio = torch.tensor(ref_audio).unsqueeze(0).float().to(self.device)
        
        # 1. Compute Mel Spectrogram (for length regulation reference)
        with torch.no_grad():
            self.target_mel = self.to_mel(ref_audio)
            
        # 2. Compute Style Embedding (CAM++)
        # Resample to 16k for CAM++
        ref_waves_16k = torchaudio.functional.resample(ref_audio, self.mel_fn_args['sampling_rate'], 16000)
        
        feat2 = torchaudio.compliance.kaldi.fbank(
            ref_waves_16k,
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000
        )
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        with torch.no_grad():
            self.target_style = self.campplus_model(feat2.unsqueeze(0))
            
        # 3. Compute Whisper Features for Target (Prompt Condition)
        # We use the whole target audio as prompt
        self.target_content = self._extract_whisper_features(ref_waves_16k)
        
        print("[Model] Target features pre-calculated.")

    def _extract_whisper_features(self, audio_16k):
        """Helper to extract Whisper features from 16kHz audio."""
        # Pad or trim if necessary, but for now assume standard processing
        inputs = self.whisper_feature_extractor(
            [audio_16k.squeeze(0).cpu().numpy()],
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=16000
        )
        input_features = self.whisper_model._mask_input_features(
            inputs.input_features, attention_mask=inputs.attention_mask
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.whisper_model.encoder(
                input_features.to(self.whisper_model.encoder.dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        
        # Downsample to match Seed-VC resolution
        features = outputs.last_hidden_state.to(torch.float32)
        features = features[:, :audio_16k.size(-1) // 320 + 1]
        return features

    @torch.no_grad()
    def process_chunk(self, audio_chunk):
        """
        Processes a single chunk of audio.
        Args:
            audio_chunk (numpy array): Input audio at config.SAMPLE_RATE
        Returns:
            numpy array: Converted audio
        """
        # 1. Prepare Input
        source_audio = torch.tensor(audio_chunk).unsqueeze(0).float().to(self.device)
        
        # 2. Resample to 16k for Whisper
        source_16k = torchaudio.functional.resample(source_audio, self.mel_fn_args['sampling_rate'], 16000)
        
        # 3. Extract Content (Whisper)
        source_content = self._extract_whisper_features(source_16k)
        
        # 4. Length Regulation
        # For real-time, we want the output length to match the input length (approximately)
        # Seed-VC uses a length regulator based on Mel length.
        
        # Calculate expected output mel length
        # Mel hop size is usually 256 or 512.
        # We can just use the source content length as a proxy for now, or force 1:1 mapping if possible.
        # However, Seed-VC expects 'ylens' (target lengths).
        
        # Let's try to map source length directly.
        # We need a dummy target length for the regulator.
        # Ideally, we want the output duration to match the input duration.
        
        # Compute Mel of source to get exact length
        source_mel = self.to_mel(source_audio)
        target_length = torch.LongTensor([int(source_mel.size(2) * self.config.LENGTH_ADJUST)]).to(self.device)
        
        # Run Length Regulator
        # Note: We pass the TARGET style/content as the "prompt" and the SOURCE content as the "input"
        
        # Source Condition (Content to be converted)
        cond, _, _, _, _ = self.model.length_regulator(
            source_content, 
            ylens=target_length,
            n_quantizers=3,
            f0=None # Assuming no F0 condition for now as per config
        )
        
        # Prompt Condition (Target Voice Style/Content)
        # We use the pre-calculated target content/mel length
        target_prompt_length = torch.LongTensor([self.target_mel.size(2)]).to(self.device)
        prompt_condition, _, _, _, _ = self.model.length_regulator(
            self.target_content,
            ylens=target_prompt_length,
            n_quantizers=3,
            f0=None
        )
        
        # Concatenate Prompt + Source
        cat_condition = torch.cat([prompt_condition, cond], dim=1)
        
        # 5. Diffusion Inference
        # We only need to generate the part corresponding to 'cond' (the source), 
        # but the model generates the whole sequence (Prompt + Source) usually?
        # Wait, the inference method in Seed-VC usually generates the target part conditioned on the prompt.
        
        # In 'inference.py':
        # vc_target = model.cfm.inference(cat_condition, ..., mel2, style2, ...)
        # mel2 is the prompt mel.
        
        with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.fp16 else torch.float32):
            vc_target = self.model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(self.device),
                self.target_mel, 
                self.target_style, 
                None, 
                self.config.DIFFUSION_STEPS,
                inference_cfg_rate=self.config.INFERENCE_CFG_RATE
            )
            
        # 6. Slice Output
        # The model output includes the prompt reconstruction? 
        # In inference.py: vc_target = vc_target[:, :, mel2.size(-1):]
        # Yes, we slice off the prompt part.
        vc_target = vc_target[:, :, self.target_mel.size(-1):]
        
        # 7. Vocoder (Mel -> Waveform)
        vc_wave = self.vocoder(vc_target.float()).squeeze()
        
        # Return as numpy array
        return vc_wave.cpu().numpy()
