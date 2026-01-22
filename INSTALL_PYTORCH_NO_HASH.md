# Alternative: Install PyTorch Without Hash Checking

If you're experiencing persistent library dependency issues, you can try installing PyTorch using `--no-deps` and `--no-cache-dir` flags, which bypasses some dependency resolution that might be causing conflicts.

## Method: Install with --no-deps and Manual Dependency Management

### Step 1: Uninstall Current PyTorch

```bash
cd /home/ailab/Desktop/innovation_research_project
source .venv/bin/activate
pip uninstall -y torch torchvision torchaudio
```

### Step 2: Install PyTorch with --no-deps

```bash
# Download the wheel (if not already downloaded)
wget https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl -O torch-2.4.0-cp310-cp310-linux_aarch64.whl

# Install without dependency checking
pip install --no-deps --no-cache-dir torch-2.4.0-cp310-cp310-linux_aarch64.whl
```

### Step 3: Install Dependencies Manually

```bash
# Install only the essential dependencies
pip install --no-cache-dir numpy==1.26.4
pip install --no-cache-dir typing-extensions filelock fsspec jinja2 sympy networkx mpmath MarkupSafe
```

### Step 4: Install TorchVision/TorchAudio with --no-deps

```bash
pip install --no-deps --no-cache-dir torchvision==0.19.0 torchaudio==2.4.0
pip install --no-cache-dir pillow
```

### Step 5: Apply All Library Fixes

```bash
./fix-all-libraries.sh
source ~/.bashrc
```

### Step 6: Test

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Why This Might Help

Using `--no-deps` prevents pip from trying to resolve dependencies automatically, which can sometimes downgrade or conflict with system libraries. You manually control what gets installed, reducing the chance of version conflicts.

## Note

This method still requires the library compatibility fixes (cuDNN, MPI, CUDA) because those are runtime library dependencies, not Python package dependencies.
