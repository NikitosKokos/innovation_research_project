import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from project.preprocessing.pipeline import ReferenceVoiceProcessor, ReferenceProcessingConfig
from project.preprocessing.comparison import ReferenceComparison

def main():
    print("=" * 60)
    print("REFERENCE VOICE PREPROCESSING DEMO")
    print("=" * 60)

    # Configuration
    # We use default config which points to audio_inputs/reference
    config = ReferenceProcessingConfig()
    
    # Ensure directories exist
    os.makedirs(config.input_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Check if there are input files
    files = [f for f in os.listdir(config.input_dir) if f.lower().endswith(('.wav', '.mp3', '.flac'))]
    if not files:
        print(f"No audio files found in {config.input_dir}")
        print("Please add some reference audio files (.wav, .mp3) to run the demo.")
        return

    # Initialize processor
    # We pass None for speaker_encoder for this demo as we don't have the specific encoder loaded here.
    # In a real scenario, you would pass the encoder function.
    processor = ReferenceVoiceProcessor(config, speaker_encoder=None)
    
    # 1. Process all references
    print("\n[Phase 1] Processing reference voices...")
    results = processor.process_all_references()
    
    # 2. Compute comparisons
    print("\n[Phase 2] Comparing raw vs. preprocessed...")
    comparison = ReferenceComparison(config)
    for filename in processor.get_reference_files():
        comp_result = comparison.compare_reference(filename)
        comparison.comparison_results.append(comp_result)
        
        if 'error' not in comp_result:
            print(f"\nFile: {filename}")
            print(f"  Raw Energy:       {comp_result['raw']['rms_energy']:.4f}")
            print(f"  Processed Energy: {comp_result['processed']['rms_energy']:.4f}")
            print(f"  STOI Similarity:  {comp_result['improvement']['stoi_preservation']:.4f}")
    
    # 3. Generate report
    print("\n[Phase 3] Generating comparison report...")
    report = comparison.generate_comparison_report()
    report_path = os.path.join(config.output_dir, "comparison_report.html")
    with open(report_path, "w") as f:
        f.write(report)
    
    print("\n[DONE] Demo complete!")
    print(f"Processed files are in: {config.output_dir}")
    print(f"Comparison report saved to: {report_path}")

if __name__ == "__main__":
    main()
