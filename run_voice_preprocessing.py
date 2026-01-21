import os
import sys
from pathlib import Path
import shutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from project.preprocessing.pipeline import ReferenceVoiceProcessor, ReferenceProcessingConfig
from project.preprocessing.comparison import ReferenceComparison

def main():
    print("=" * 60)
    print("VOICE PREPROCESSING PIPELINE")
    print("=" * 60)

    input_dir = "audio_inputs/voice_preprocessing"
    output_dir = "audio_outputs/voice_preprocessed"
    
    # Create input directory if it doesn't exist
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Created input directory: {input_dir}")
        print("Please put your audio files there and run this script again.")
        return

    # Check for files
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))]
    if not files:
        print(f"No audio files found in {input_dir}")
        return

    # Configure the pipeline
    config = ReferenceProcessingConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        embedding_dir=os.path.join(output_dir, "embeddings"),
        denoising_method="noisereduce", # Explicitly use the improved method
        compute_metrics=True
    )
    
    # Initialize processor
    processor = ReferenceVoiceProcessor(config, speaker_encoder=None)
    
    print(f"\nFound {len(files)} files. Starting processing...")
    print(f"Using denoising method: {config.denoising_method}")
    
    # Process
    results = processor.process_all_references()
    
    # Comparison
    print("\nGenerating quality comparison report...")
    comparison = ReferenceComparison(config)
    
    # We need to make sure the processor actually processed files before comparing
    processed_files = processor.get_reference_files()
    
    for filename in processed_files:
        try:
            comp_result = comparison.compare_reference(filename)
            comparison.comparison_results.append(comp_result)
            
            if 'error' not in comp_result:
                print(f"\nFile: {filename}")
                print(f"  Processed Energy: {comp_result['processed']['rms_energy']:.4f}")
                print(f"  STOI Similarity:  {comp_result['improvement']['stoi_preservation']:.4f}")
        except Exception as e:
            print(f"Skipping comparison for {filename}: {e}")

    # Generate report
    report = comparison.generate_comparison_report()
    report_path = os.path.join(output_dir, "processing_report.html")
    with open(report_path, "w") as f:
        f.write(report)
        
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Input Directory:  {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Report:           {report_path}")

if __name__ == "__main__":
    main()
