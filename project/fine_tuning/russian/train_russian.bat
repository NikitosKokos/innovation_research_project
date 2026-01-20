@echo off
echo ==================================================
echo      Checking GPU Availability...
echo ==================================================
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Count: {torch.cuda.device_count()}')"
echo.

echo ==================================================
echo      Seed-VC: Russian Language Fine-tuning (Optimized)
echo ==================================================
echo.

REM --- Configuration ---
REM Increased batch size and workers for speed.
REM If you get "Out of Memory", reduce --batch-size to 4.
REM If you get DataLoader errors, reduce --num-workers to 0.

python train.py ^
    --config "configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml" ^
    --pretrained-ckpt "checkpoints/models--Plachta--Seed-VC/snapshots/257283f9f41585055e8f858fba4fd044e5caed6e/DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth" ^
    --dataset-dir "datasets/russian_finetune" ^
    --run-name "russian_finetune_small_optimized" ^
    --batch-size 4 ^
    --num-workers 2 ^
    --max-epochs 20 ^
    --max-steps 5000 ^
    --save-every 500 ^
    --gpu 0

echo.
echo ==================================================
echo Fine-tuning Finished!
echo Checkpoint: runs/russian_finetune_small_optimized/ft_model.pth
echo ==================================================
pause
