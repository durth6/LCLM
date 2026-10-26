#!/usr/bin/env bash
# Train three selector seeds on one fixed document-level split.
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --array=0-2
#SBATCH --time=02:00:00
#SBATCH --job-name=lclm-p1-3seed
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-p1-3seed-%A_%a.log

set -euo pipefail

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly python_bin=/data/home/scxj745/.conda/envs/qwen35/bin/python
readonly checkpoint=${root}/artifacts/models/0.6b-4b-LCLM-8x
readonly train_dir=${root}/artifacts/data/selector_train_mixed_4to32k
readonly contribution_jsonl=${train_dir}/contributions-zgrad-full/contributions-zgrad-full.jsonl
readonly split_seed=${SPLIT_SEED:-6201}
readonly seeds=(6201 6202 6203)
readonly task_index=${SLURM_ARRAY_TASK_ID:-0}
readonly seed=${seeds[task_index]}
readonly selector_checkpoint=${root}/artifacts/checkpoints/selector-p1-docsplit-posfix-s${seed}

test -n "${seed:-}"
test -f "${train_dir}/cache/manifest.json"
test -f "${contribution_jsonl}"
cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

exec "${python_bin}" -u -m train.train_selector \
  --checkpoint "${checkpoint}" \
  --cache-dir "${train_dir}/cache" \
  --queries-jsonl "${train_dir}/queries.jsonl" \
  --oracle-jsonl "${train_dir}/oracle_latents.jsonl" \
  --contribution-jsonl "${contribution_jsonl}" \
  --out "${selector_checkpoint}" \
  --epochs 5 \
  --learning-rate 3e-4 \
  --rank-weight 1.0 \
  --bce-weight 1.0 \
  --contribution-weight 1.0 \
  --contribution-temperature 1.0 \
  --contribution-normalization zgrad \
  --validation-fraction 0.125 \
  --eval-selection-ratio 0.25 \
  --split-seed "${split_seed}" \
  --seed "${seed}" \
  --dtype bf16 \
  --device cuda
