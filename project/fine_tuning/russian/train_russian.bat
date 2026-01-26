@echo off
echo ==================================================
echo      Checking GPU Availability...
echo ==================================================
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Count: {torch.cuda.device_count()}')"
echo.

echo ==================================================
echo      Seed-VC: Russian Language Fine-tuning (Resume)
echo ==================================================
echo.

REM --- Configuration ---
REM Resuming training from existing checkpoint in runs/russian_finetune_small_v3
REM The script will automatically find the latest checkpoint (DiT_epoch_*_step_*.pth)
REM If you get "Out of Memory", reduce --batch-size to 4.
REM If you get DataLoader errors, reduce --num-workers to 0.

python train.py ^
    --config "configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml" ^
    --dataset-dir "datasets/russian_finetune" ^
    --run-name "russian_finetune_small_v3" ^
    --pretrained-ckpt "runs/russian_finetune_small_v3/ft_model.pth" ^
    --batch-size 4 ^
    --num-workers 2 ^
    --max-epochs 100 ^
    --max-steps 50000 ^
    --save-every 500 ^
    --gpu 0

echo.
echo ==================================================
echo Fine-tuning Finished!
echo Checkpoints saved to: runs/russian_finetune_small_v3/
echo ==================================================
pause
