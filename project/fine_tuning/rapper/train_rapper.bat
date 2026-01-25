@echo off
echo ==================================================
echo      Seed-VC: Rapper (Oxxxymiron) Fine-tuning
echo ==================================================
echo.
echo Base Model: runs/russian_finetune_small_v3/ft_model.pth
echo Dataset: datasets/rapper_finetune
echo GPU: RTX 4060 (Optimized)
echo.

REM Run from project root
python train.py ^
    --config "./checkpoints\models--Plachta--Seed-VC\snapshots\257283f9f41585055e8f858fba4fd044e5caed6e\config_dit_mel_seed_uvit_whisper_small_wavenet.yml" ^
    --pretrained-ckpt "runs/russian_finetune_small_v3/ft_model.pth" ^
    --dataset-dir "datasets/rapper_finetune" ^
    --run-name "rapper_oxxxy_finetune" ^
    --batch-size 4 ^
    --max-epochs 100 ^
    --save-every 200 ^
    --num-workers 2

echo.
echo Training Finished!
echo Checkpoint: runs/rapper_oxxxy_finetune/ft_model.pth
pause
