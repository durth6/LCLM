#!/usr/bin/env bash
# Evaluate the full P1 selector on 4K/8K/16K/32K RULER smoke splits.
# Submit after training with: sbatch --array=0-3 --gpus=1 -p gpu_4090 ...
#
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --job-name=lclm-p1-eval
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-p1-eval-%A_%a.log

set -euo pipefail

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly checkpoint="${CHECKPOINT:-${root}/artifacts/models/0.6b-4b-LCLM-8x}"
readonly selector_checkpoint="${SELECTOR_CHECKPOINT:-${root}/artifacts/checkpoints/selector-stage-b-contribution-full-s6201}"
readonly seed="${SEED:-6201}"
readonly task_index="${SLURM_ARRAY_TASK_ID:-0}"
readonly python_bin="${PYTHON_BIN:-/data/home/scxj745/.conda/envs/qwen35/bin/python}"

case "${task_index}" in
  0) tag=4k; input_dir="${root}/artifacts/data/ruler_niah_4k_smoke"; cache_dir="${root}/artifacts/runs/smoke-k256/cache" ;;
  1) tag=8k; input_dir="${root}/artifacts/data/ruler_niah_8k_smoke"; cache_dir="${root}/artifacts/runs/ruler-8k-cache" ;;
  2) tag=16k; input_dir="${root}/artifacts/data/ruler_niah_16k_smoke"; cache_dir="${root}/artifacts/runs/ruler-16k-cache" ;;
  3) tag=32k; input_dir="${root}/artifacts/data/ruler_niah_32k_smoke"; cache_dir="${root}/artifacts/runs/ruler-32k-cache" ;;
  *) echo "unsupported SLURM_ARRAY_TASK_ID=${task_index}" >&2; exit 2 ;;
esac

test -f "${checkpoint}/.download_complete"
test -f "${selector_checkpoint}/selector.safetensors"
test -f "${input_dir}/oracle_latents.jsonl"
test -f "${cache_dir}/manifest.json"
cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${python_bin}" -u -m inference.examples.eval_query_aware_niah \
  --checkpoint "${checkpoint}" \
  --input-dir "${input_dir}" \
  --cache-dir "${cache_dir}" \
  --work-dir "${root}/artifacts/runs/p1-contribution-full-25pct-ctx${tag}-s${seed}" \
  --selection-ratio 0.25 \
  --selector-mode learned \
  --selector-checkpoint "${selector_checkpoint}" \
  --oracle-jsonl "${input_dir}/oracle_latents.jsonl" \
  --max-tokens 128 \
  --dtype bf16 \
  --random-seed "${seed}" \
  --reuse-queries 1 \
  --repeat-each-query 1 \
  --max-resident-caches 1
