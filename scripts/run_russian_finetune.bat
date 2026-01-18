@echo off
echo Starting Fine-tuning for Russian Voice...

:: 1. Prepare Dataset
python scripts/prepare_russian_dataset.py

:: 2. Run Training
:: config: The real-time tiny model config
:: run-name: The name of the output folder in ./runs/
:: max-epochs: 100 is enough for fine-tuning on small data
:: save-every: Save checkpoint frequently
:: pretrained-model: Explicitly use the tiny model

echo.
echo Running training script...
python train.py ^
    --config configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml ^
    --dataset-dir datasets/russian_finetune ^
    --run-name russian_finetune_v1 ^
    --batch-size 4 ^
    --max-epochs 20 ^
    --save-every 10 ^
    --num-workers 0

echo.
echo Training complete! Check ./runs/russian_finetune_v1/ for checkpoints.
pause
