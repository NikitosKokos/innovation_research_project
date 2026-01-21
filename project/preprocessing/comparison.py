import os
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, List
from project.preprocessing.pipeline import ReferenceProcessingConfig
from project.preprocessing.evaluation import compute_pesq, compute_stoi

class ReferenceComparison:
    """
    Compare raw vs. preprocessed reference quality
    """
    
    def __init__(self, config: ReferenceProcessingConfig):
        self.config = config
        self.comparison_results = []
    
    def compare_reference(self, filename: str) -> Dict:
        """
        Compare raw and preprocessed version of reference
        """
        raw_path = os.path.join(self.config.input_dir, filename)
        processed_path = os.path.join(
            self.config.output_dir,
            f"{Path(filename).stem}_processed{Path(filename).suffix}"
        )
        
        if not os.path.exists(processed_path):
             return {
                'filename': filename,
                'error': 'Processed file not found'
            }
        
        # Load both versions
        try:
            raw_audio, sr = librosa.load(raw_path, sr=self.config.sample_rate)
            processed_audio, _ = librosa.load(processed_path, sr=self.config.sample_rate)
        except Exception as e:
             return {
                'filename': filename,
                'error': f'Error loading audio: {e}'
            }
        
        # Ensure same length
        min_len = min(len(raw_audio), len(processed_audio))
        raw_audio = raw_audio[:min_len]
        processed_audio = processed_audio[:min_len]
        
        comparison = {
            'filename': filename,
            'raw': {},
            'processed': {},
            'improvement': {}
        }
        
        # Compute PESQ (requires 16kHz)
        # Note: comparison with itself is perfect (4.5), but we want to see how much we deviated 
        # OR we want to compare against a "ground truth" clean version if we had one.
        # Here we compare processed against raw to see "degradation" or "change".
        # But wait, the guide says:
        # raw_pesq = compute_pesq(raw_16k, raw_16k) -> 4.5
        # proc_pesq = compute_pesq(raw_16k, proc_16k) -> ?
        # This measures similarity to raw. If raw is noisy, and processed is clean, PESQ might be low.
        # But for "quality", we usually need a clean reference.
        # The guide implementation does this, so I will follow.
        
        raw_16k = librosa.resample(raw_audio, orig_sr=sr, target_sr=16000)
        proc_16k = librosa.resample(processed_audio, orig_sr=sr, target_sr=16000)
        
        raw_pesq = compute_pesq(raw_16k, raw_16k, sr=16000)  # Self-reference
        proc_pesq = compute_pesq(raw_16k, proc_16k, sr=16000)  # Against raw
        
        comparison['raw']['pesq_self'] = raw_pesq
        comparison['processed']['pesq_vs_raw'] = proc_pesq
        
        # STOI (intelligibility)
        raw_stoi = compute_stoi(raw_audio, raw_audio, sr=sr)
        proc_stoi = compute_stoi(raw_audio, processed_audio, sr=sr)
        
        comparison['raw']['stoi_self'] = raw_stoi
        comparison['processed']['stoi_vs_raw'] = proc_stoi
        comparison['improvement']['stoi_preservation'] = proc_stoi
        
        # Energy metrics
        comparison['raw']['rms_energy'] = float(np.sqrt(np.mean(raw_audio ** 2)))
        comparison['processed']['rms_energy'] = float(np.sqrt(np.mean(processed_audio ** 2)))
        
        return comparison
    
    def generate_comparison_report(self) -> str:
        """Generate HTML report comparing raw vs. preprocessed reference"""
        report = """
        <html>
        <head>
            <title>Reference Voice Preprocessing Comparison</title>
            <style>
                body { font-family: Arial; margin: 20px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid black; padding: 10px; text-align: left; }
                th { background-color: #f0f0f0; }
                .improvement { color: green; font-weight: bold; }
                .error { color: red; }
            </style>
        </head>
        <body>
            <h1>Reference Voice Preprocessing Comparison</h1>
            <table>
                <tr>
                    <th>File</th>
                    <th>Raw PESQ (Self)</th>
                    <th>Processed vs Raw PESQ</th>
                    <th>STOI Preservation</th>
                    <th>RMS Energy (Raw)</th>
                    <th>RMS Energy (Processed)</th>
                </tr>
        """
        
        for result in self.comparison_results:
            if 'error' in result:
                 report += f"""
                <tr>
                    <td>{result['filename']}</td>
                    <td colspan="5" class="error">{result['error']}</td>
                </tr>
                """
                 continue

            report += f"""
                <tr>
                    <td>{result['filename']}</td>
                    <td>{result['raw'].get('pesq_self', 'N/A'):.2f}</td>
                    <td>{result['processed'].get('pesq_vs_raw', 'N/A'):.2f}</td>
                    <td class="improvement">{result['improvement'].get('stoi_preservation', 0):.3f}</td>
                    <td>{result['raw']['rms_energy']:.4f}</td>
                    <td>{result['processed']['rms_energy']:.4f}</td>
                </tr>
            """
        
        report += """
            </table>
        </body>
        </html>
        """
        
        return report
