#!/bin/bash
#SBATCH --gpus=1
#SBATCH -p gpu_4090
#SBATCH -o artifacts/logs/neighbor-expansion-test-%j.log

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

PYTHON=/data/home/scxj745/.conda/envs/qwen35/bin/python
CHECKPOINT=artifacts/checkpoints/selector-p1-docsplit-posfix-s6201
INPUT_DIR=artifacts/data/ruler_all_tasks_v2_16k

# Test 1: radius=2, ratio=0.20
$PYTHON -u inference/examples/eval_query_aware_niah.py \
  --checkpoint artifacts/models/0.6b-4b-LCLM-8x \
  --selector-checkpoint $CHECKPOINT \
  --input-dir $INPUT_DIR \
  --work-dir artifacts/runs/test-neighbor-r2-ratio20-16k-s6201 \
  --selection-ratio 0.20 \
  --selector-mode learned \
  --neighbor-radius 2 \
  --position-policy full_layout \
  --max-examples 104 \
  --device cuda

# Test 2: radius=3, ratio=0.15
$PYTHON -u inference/examples/eval_query_aware_niah.py \
  --checkpoint artifacts/models/0.6b-4b-LCLM-8x \
  --selector-checkpoint $CHECKPOINT \
  --input-dir $INPUT_DIR \
  --work-dir artifacts/runs/test-neighbor-r3-ratio15-16k-s6201 \
  --selection-ratio 0.15 \
  --selector-mode learned \
  --neighbor-radius 3 \
  --position-policy full_layout \
  --max-examples 104 \
  --device cuda

echo "Neighbor expansion tests completed"
