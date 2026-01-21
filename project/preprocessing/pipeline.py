import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import librosa
import soundfile as sf
from dataclasses import dataclass, field
from datetime import datetime

# Adjust imports to local project structure
from project.preprocessing.denoising import get_denoiser
from project.preprocessing.normalization import LUFSNormalizer
from project.preprocessing.compression import DynamicRangeCompressor
from project.preprocessing.evaluation import compute_pesq, compute_stoi

@dataclass
class ReferenceProcessingConfig:
    """Configuration for reference voice processing"""
    input_dir: str = "audio_inputs/reference"
    output_dir: str = "audio_outputs/reference_processed"
    embedding_dir: str = "audio_outputs/reference_processed/embeddings"
    
    # Preprocessing parameters (optimized for reference)
    sample_rate: int = 22050
    denoising_method: str = "noisereduce"  # Changed from "wiener" to "noisereduce" for better quality
    target_lufs: float = -23.0
    compression_threshold: float = -15.0  # Slightly higher for reference
    compression_ratio: float = 3.0
    compression_attack_ms: float = 10.0
    compression_release_ms: float = 150.0
    
    # Reference-specific settings
    batch_size: int = 4  # Can process multiple at once
    cache_embeddings: bool = True
    compute_metrics: bool = True


class ReferenceVoiceProcessor:
    """
    Process rapper reference voices in batch for optimal speaker embeddings
    """
    
    def __init__(self, config: ReferenceProcessingConfig, speaker_encoder=None):
        self.config = config
        self.speaker_encoder = speaker_encoder
        
        # Initialize preprocessing components
        self.denoiser = get_denoiser(config.denoising_method, sr=config.sample_rate)
        self.normalizer = LUFSNormalizer(target_lufs=config.target_lufs, 
                                         sr=config.sample_rate)
        self.compressor = DynamicRangeCompressor(
            threshold_db=config.compression_threshold,
            ratio=config.compression_ratio,
            attack_ms=config.compression_attack_ms,
            release_ms=config.compression_release_ms,
            sr=config.sample_rate
        )
        
        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.embedding_dir, exist_ok=True)
        
        # Processing log
        self.processing_log = {
            'timestamp': datetime.now().isoformat(),
            'config': config.__dict__,
            'processed_files': [],
            'statistics': {}
        }
    
    def get_reference_files(self) -> List[str]:
        """Get all audio files from reference directory"""
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a'}
        if not os.path.exists(self.config.input_dir):
            os.makedirs(self.config.input_dir, exist_ok=True)
            return []
            
        files = [f for f in os.listdir(self.config.input_dir)
                if Path(f).suffix.lower() in audio_extensions]
        return sorted(files)
    
    def process_single_reference(self, filename: str) -> Dict:
        """
        Process single reference voice file
        
        Returns:
            Dictionary with processing results and metrics
        """
        input_path = os.path.join(self.config.input_dir, filename)
        output_filename = f"{Path(filename).stem}_processed{Path(filename).suffix}"
        output_path = os.path.join(self.config.output_dir, output_filename)
        
        # Load audio
        try:
            audio, sr = librosa.load(input_path, sr=self.config.sample_rate)
        except Exception as e:
            return {'filename': filename, 'status': 'error', 'error': f"Failed to load: {e}"}
            
        original_audio = audio.copy()
        
        result = {
            'filename': filename,
            'duration_sec': len(audio) / self.config.sample_rate,
            'processing_steps': []
        }
        
        try:
            # Step 1: Denoising
            # print(f"Denoising {filename}...")
            audio = self.denoiser.denoise(audio)
            result['processing_steps'].append('denoised')
            
            # Step 2: LUFS Normalization
            # print(f"Normalizing {filename}...")
            audio = self.normalizer.normalize(audio)
            result['processing_steps'].append('normalized')
            
            # Step 3: Dynamic Range Compression
            # print(f"Compressing {filename}...")
            audio = self.compressor.compress(audio)
            result['processing_steps'].append('compressed')
            
            # Save processed audio
            sf.write(output_path, audio, self.config.sample_rate)
            result['output_path'] = output_path
            result['saved'] = True
            
            # Compute metrics (if original reference available for comparison)
            if self.config.compute_metrics:
                metrics = self._compute_preprocessing_metrics(original_audio, audio)
                result['metrics'] = metrics
            
            # Extract speaker embedding
            if self.speaker_encoder is not None:
                # Assuming speaker_encoder is a callable that takes audio
                try:
                    embedding = self.speaker_encoder(audio)
                    embedding_path = os.path.join(
                        self.config.embedding_dir,
                        f"{Path(filename).stem}_embedding.npy"
                    )
                    np.save(embedding_path, embedding)
                    result['embedding_path'] = embedding_path
                    result['embedding_shape'] = embedding.shape
                except Exception as e:
                    print(f"Embedding extraction failed: {e}")
            
            result['status'] = 'success'
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            print(f"Error processing {filename}: {e}")
        
        return result
    
    def _compute_preprocessing_metrics(self, original: np.ndarray, 
                                       processed: np.ndarray) -> Dict:
        """Compute metrics showing preprocessing impact"""
        metrics = {}
        
        # Energy comparison
        original_energy = np.sqrt(np.mean(original ** 2))
        processed_energy = np.sqrt(np.mean(processed ** 2))
        metrics['original_energy'] = float(original_energy)
        metrics['processed_energy'] = float(processed_energy)
        
        if original_energy > 0:
            metrics['energy_change_db'] = 20 * np.log10(processed_energy / (original_energy + 1e-10))
        else:
            metrics['energy_change_db'] = 0.0
        
        # Dynamic range
        if original_energy > 0:
            metrics['original_dynamic_range_db'] = 20 * np.log10(
                np.max(np.abs(original)) / (original_energy + 1e-10)
            )
        else:
            metrics['original_dynamic_range_db'] = 0.0
            
        if processed_energy > 0:
            metrics['processed_dynamic_range_db'] = 20 * np.log10(
                np.max(np.abs(processed)) / (processed_energy + 1e-10)
            )
        else:
             metrics['processed_dynamic_range_db'] = 0.0
        
        # Spectral analysis
        original_spectrum = np.abs(np.fft.rfft(original))
        processed_spectrum = np.abs(np.fft.rfft(processed))
        
        if np.mean(original_spectrum) > 0:
            metrics['spectral_flatness_original'] = float(
                np.exp(np.mean(np.log(original_spectrum + 1e-10))) / 
                (np.mean(original_spectrum) + 1e-10)
            )
        else:
            metrics['spectral_flatness_original'] = 0.0

        if np.mean(processed_spectrum) > 0:
            metrics['spectral_flatness_processed'] = float(
                np.exp(np.mean(np.log(processed_spectrum + 1e-10))) / 
                (np.mean(processed_spectrum) + 1e-10)
            )
        else:
             metrics['spectral_flatness_processed'] = 0.0
        
        return metrics
    
    def process_all_references(self) -> Dict:
        """
        Process all reference voices in batch
        
        Returns:
            Summary of all processing results
        """
        files = self.get_reference_files()
        print(f"Found {len(files)} reference files to process")
        
        results = []
        for i, filename in enumerate(files):
            print(f"[{i+1}/{len(files)}] Processing {filename}...")
            result = self.process_single_reference(filename)
            results.append(result)
            self.processing_log['processed_files'].append(result)
        
        # Compute statistics
        successful = len([r for r in results if r['status'] == 'success'])
        self.processing_log['statistics'] = {
            'total_files': len(files),
            'successful': successful,
            'failed': len(files) - successful,
            'success_rate': successful / len(files) if files else 0
        }
        
        # Save processing log
        log_path = os.path.join(self.config.output_dir, 'processing_log.json')
        with open(log_path, 'w') as f:
            json.dump(self.processing_log, f, indent=2, default=str)
        
        print(f"\nProcessing complete!")
        print(f"Success rate: {successful}/{len(files)}")
        print(f"Log saved to: {log_path}")
        
        return {
            'results': results,
            'summary': self.processing_log['statistics']
        }
