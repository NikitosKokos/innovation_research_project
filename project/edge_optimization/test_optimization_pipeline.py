import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from project.edge_optimization.find_optimal_config import main as run_optimization

def test_pipeline():
    print("\n--- Running Edge Optimization Pipeline Test ---")

    # Define paths relative to project root (assuming script runs from project root or edge_optimization dir)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    checkpoint_path = os.path.join(project_root, 'AI_models', 'rapper_oxxxy_finetune', 'ft_model.pth')
    config_path = os.path.join(project_root, 'AI_models', 'rapper_oxxxy_finetune', 'config_dit_mel_seed_uvit_whisper_small_wavenet.yml')
    test_audio_path = os.path.join(project_root, 'audio_inputs', 'user', 'test02.wav')
    reference_audio_path = os.path.join(project_root, 'audio_inputs', 'reference', 'ref01_processed.wav')
    output_json_path = os.path.join(project_root, 'project', 'edge_optimization', 'test_edge_optimization_results.json')
    best_model_dir = os.path.join(project_root, 'AI_models', 'best')

    # Ensure dummy audio files exist for testing
    if not os.path.exists(test_audio_path):
        print(f"Warning: Test audio not found at {test_audio_path}. Creating a dummy file.")
        import soundfile as sf
        import numpy as np
        sf.write(test_audio_path, np.random.randn(16000 * 5), 16000) # 5 seconds of noise
    if not os.path.exists(reference_audio_path):
        print(f"Warning: Reference audio not found at {reference_audio_path}. Creating a dummy file.")
        import soundfile as sf
        import numpy as np
        sf.write(reference_audio_path, np.random.randn(16000 * 5), 16000) # 5 seconds of noise


    # Prepare arguments for run_optimization
    sys.argv = [
        "find_optimal_config.py", # Script name (ignored by argparse)
        "--checkpoint", checkpoint_path,
        "--config", config_path,
        "--test-audio", test_audio_path,
        "--reference-audio", reference_audio_path,
        "--device", "cpu", # Use CPU for initial testing to avoid CUDA/TensorRT issues on all systems
        "--output", output_json_path,
        "--block-time", "0.12",
        "--extra-context-left", "1.5",
        "--extra-context-right", "0.02",
        "--diffusion-steps", "10"
    ]

    # Run the optimization
    run_optimization()

    print("\n--- Optimization Pipeline Test Complete ---")

    # Optionally, read and print the results from the JSON file
    if os.path.exists(output_json_path):
        with open(output_json_path, 'r') as f:
            results = json.load(f)
        print("Optimization Results Summary:")
        print(json.dumps(results['optimal_config'], indent=2))
        print(f"Best model should be saved in: {best_model_dir}")

if __name__ == "__main__":
    # Ensure necessary libraries are installed for ONNX Runtime
    # This part is for local testing, not for the agent to execute
    # try:
    #     import onnxruntime as rt
    #     print("ONNX Runtime installed.")
    # except ImportError:
    #     print("ONNX Runtime not found. Please install with: pip install onnxruntime")
    #     sys.exit(1)

    test_pipeline()
