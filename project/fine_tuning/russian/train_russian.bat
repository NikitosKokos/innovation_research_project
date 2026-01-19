@echo off
echo ==================================================
echo      Seed-VC: Russian Language Fine-tuning
echo ==================================================
echo.
echo Base Model: ./checkpoints\models--Plachta--Seed-VC\snapshots\257283f9f41585055e8f858fba4fd044e5caed6e\DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth
echo Dataset: datasets/russian_finetune
echo.

REM Run from project root
python train.py ^
    --config "./checkpoints\models--Plachta--Seed-VC\snapshots\257283f9f41585055e8f858fba4fd044e5caed6e\config_dit_mel_seed_uvit_whisper_small_wavenet.yml" ^
    --pretrained-ckpt "./checkpoints\models--Plachta--Seed-VC\snapshots\257283f9f41585055e8f858fba4fd044e5caed6e\DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth" ^
    --dataset-dir "datasets/russian_finetune" ^
    --run-name "russian_finetune_small_v3" ^
    --batch-size 4 ^
    --max-epochs 20 ^
    --save-every 500 ^
    --num-workers 0

echo.
echo Training Finished!
echo Checkpoint: runs/russian_finetune_small/ft_model.pth
pause
