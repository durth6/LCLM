#!/bin/bash
#SBATCH --gpus=1
#SBATCH -p gpu_4090
#SBATCH -o artifacts/logs/slot-policy-test-%j.log

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

PYTHON=/data/home/scxj745/.conda/envs/qwen35/bin/python
CHECKPOINT=artifacts/checkpoints/selector-p1-docsplit-posfix-s6201
INPUT_DIR=artifacts/data/ruler_all_tasks_v2_16k

# Test entity slot policy (for multi-key queries)
$PYTHON -u inference/examples/eval_query_aware_niah.py \
  --checkpoint artifacts/models/0.6b-4b-LCLM-8x \
  --selector-checkpoint $CHECKPOINT \
  --input-dir $INPUT_DIR \
  --work-dir artifacts/runs/test-entity-slots-16k-s6201 \
  --selection-ratio 0.25 \
  --selector-mode learned \
  --slot-policy entity \
  --slot-max-slots 8 \
  --slot-combine union \
  --position-policy full_layout \
  --max-examples 104 \
  --device cuda

echo "Slot policy test completed"
