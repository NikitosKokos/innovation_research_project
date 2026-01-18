@echo off
echo ==================================================
echo      Seed-VC Russian Fine-tuning Launcher
echo ==================================================

:: 1. Prepare Data
echo.
echo [Step 1/2] Processing dataset...
python scripts/prepare_local_dataset.py

:: 2. Run Training
echo.
echo [Step 2/2] Starting Training...
echo Parameters:
echo - Model: seed-uvit-tat-xlsr-tiny (Real-time optimized)
echo - Batch Size: 4
echo - Epochs: 50
echo.

:: Check for Conda Python to ensure CUDA support
set PYTHON_EXE=python
if exist "C:\Users\nikit\Miniconda3\python.exe" (
    echo Using Miniconda Python: C:\Users\nikit\Miniconda3\python.exe
    set PYTHON_EXE="C:\Users\nikit\Miniconda3\python.exe"
)

%PYTHON_EXE% train.py ^
    --config configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml ^
    --dataset-dir datasets/russian_finetune ^
    --run-name russian_finetune_v2 ^
    --batch-size 4 ^
    --max-epochs 50 ^
    --save-every 10 ^
    --num-workers 0

echo.
echo ==================================================
echo Training Finished!
echo Checkpoints are located in: runs/russian_finetune_v2/
echo ==================================================
pause
