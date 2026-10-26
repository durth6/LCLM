#!/usr/bin/env bash
# Example:
#   SELECTOR_TRAIN_DIR=/abs/path/to/independent_selector_train \
#     sbatch scripts/run_train_selector_stage_a.sbatch
#
# SELECTOR_TRAIN_DIR must be an independently generated training split, never
# an official RULER/LongBench evaluation directory.
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=lclm-selector-a
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-selector-a-%j.log

set -euo pipefail

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly checkpoint="${root}/artifacts/models/0.6b-4b-LCLM-8x"
readonly train_dir="${SELECTOR_TRAIN_DIR:?set SELECTOR_TRAIN_DIR to an independent training split}"
readonly cache_dir="${SELECTOR_CACHE_DIR:-${train_dir}/cache}"
readonly output_dir="${SELECTOR_OUTPUT_DIR:-${root}/artifacts/checkpoints/selector-stage-a}"

test -f "${checkpoint}/.download_complete"
test -f "${cache_dir}/manifest.json"
test -f "${train_dir}/queries.jsonl"
test -f "${train_dir}/oracle_latents.jsonl"

cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
exec /data/home/scxj745/.conda/envs/qwen35/bin/python -u -m train.train_selector \
  --checkpoint "${checkpoint}" \
  --cache-dir "${cache_dir}" \
  --queries-jsonl "${train_dir}/queries.jsonl" \
  --oracle-jsonl "${train_dir}/oracle_latents.jsonl" \
  --out "${output_dir}" \
  --retrieval-dim 128 \
  --epochs "${EPOCHS:-10}" \
  --learning-rate "${LEARNING_RATE:-3e-4}" \
  --validation-fraction "${VALIDATION_FRACTION:-0.1}" \
  --eval-selection-ratio "${EVAL_SELECTION_RATIO:-0.25}" \
  --seed "${SEED:-4231}" \
  --dtype bf16
