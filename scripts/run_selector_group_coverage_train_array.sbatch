#!/usr/bin/env bash
# Train three combination-aware selector seeds with explicit per-key group
# coverage losses.  The utility and role sidecars are offline labels only.
# The answer-time evaluator never reads these files.
#
# Submit on the 4090 queue:
#   sbatch --array=0-2 --gpus=1 -p gpu_4090 \
#     scripts/run_selector_group_coverage_train_array.sbatch
#
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --array=0-2
#SBATCH --time=04:00:00
#SBATCH --job-name=lclm-group-selector
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-group-selector-%A_%a.log

set -euo pipefail

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly python_bin="${PYTHON_BIN:-/data/home/scxj745/.conda/envs/qwen35/bin/python}"
readonly checkpoint="${CHECKPOINT:-${root}/artifacts/models/0.6b-4b-LCLM-8x}"
readonly train_dir="${TRAIN_DIR:-${root}/artifacts/data/selector_train_mixed_4to32k}"
readonly utility_jsonl="${UTILITY_JSONL:-${train_dir}/utility-oracle-full-layout-v2/utility-oracle-full-layout-v2.jsonl}"
readonly split_seed="${SPLIT_SEED:-6201}"
readonly task_index="${SLURM_ARRAY_TASK_ID:-0}"
readonly utility_weight="${UTILITY_WEIGHT:-0.5}"
readonly role_weight="${ROLE_COVERAGE_WEIGHT:-0.5}"
readonly key_answer_group_weight="${KEY_ANSWER_GROUP_WEIGHT:-0.5}"
readonly support_group_weight="${SUPPORT_GROUP_WEIGHT:-0.5}"
readonly group_temperature="${GROUP_COVERAGE_TEMPERATURE:-1.0}"
readonly group_margin="${GROUP_MAX_MARGIN:-0.0}"
readonly group_ratio="${GROUP_SELECTION_RATIO:-0.25}"
readonly balance_weight="${POSITION_BALANCE_WEIGHT:-0.0}"
readonly epochs="${EPOCHS:-5}"
readonly seeds=(6201 6202 6203)

if (( task_index < 0 || task_index >= ${#seeds[@]} )); then
  echo "unsupported SLURM_ARRAY_TASK_ID=${task_index}" >&2
  exit 2
fi
readonly seed="${seeds[task_index]}"
readonly selector_checkpoint="${SELECTOR_CHECKPOINT:-${root}/artifacts/checkpoints/selector-group-coverage-utility-s${seed}}"

test -f "${checkpoint}/.download_complete"
test -f "${train_dir}/cache/manifest.json"
test -f "${train_dir}/queries.jsonl"
test -f "${train_dir}/oracle_latents_roles.jsonl"
test -f "${utility_jsonl}"

cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

exec "${python_bin}" -u -m train.train_selector \
  --checkpoint "${checkpoint}" \
  --cache-dir "${train_dir}/cache" \
  --queries-jsonl "${train_dir}/queries.jsonl" \
  --oracle-jsonl "${train_dir}/oracle_latents_roles.jsonl" \
  --utility-jsonl "${utility_jsonl}" \
  --out "${selector_checkpoint}" \
  --epochs "${epochs}" \
  --learning-rate 3e-4 \
  --rank-weight 1.0 \
  --bce-weight 1.0 \
  --utility-weight "${utility_weight}" \
  --role-coverage-weight "${role_weight}" \
  --key-answer-group-weight "${key_answer_group_weight}" \
  --support-group-weight "${support_group_weight}" \
  --group-coverage-temperature "${group_temperature}" \
  --group-max-margin "${group_margin}" \
  --group-selection-ratio "${group_ratio}" \
  --position-balance-weight "${balance_weight}" \
  --role-temperature 1.0 \
  --position-balance-buckets 8 \
  --validation-fraction 0.125 \
  --eval-selection-ratio 0.25 \
  --split-seed "${split_seed}" \
  --seed "${seed}" \
  --dtype bf16 \
  --device cuda
