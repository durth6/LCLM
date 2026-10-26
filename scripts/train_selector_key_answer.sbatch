#!/bin/bash
#SBATCH --gpus=1
#SBATCH -p gpu_4090
#SBATCH -o artifacts/logs/train-selector-key-answer-%j.log

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TMPDIR=/data/run01/scxj745/context-exp/LCLM/.tmp

PYTHON=/data/home/scxj745/.conda/envs/qwen35/bin/python

# 使用 key_answer role 训练 (平均 7.26 latents vs 2.88)
$PYTHON -u train/train_selector.py \
  --checkpoint artifacts/models/0.6b-4b-LCLM-8x \
  --train-data artifacts/data/selector_train_mixed_4to32k \
  --oracle-role key_answer \
  --train-steps 1120 \
  --save-steps 224 \
  --learning-rate 1e-4 \
  --warmup-ratio 0.1 \
  --weight-decay 0.01 \
  --random-seed 6201 \
  --split-seed 6200 \
  --k-decoder-ratio 0.25 \
  --output-dir artifacts/checkpoints/selector-key-answer-s6201 \
  --device cuda

echo "Training with key_answer oracle completed"
