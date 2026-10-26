#!/usr/bin/env bash
# Evaluate combination-aware selector seeds on the corrected v2 13-task data.
# Each array task evaluates one seed over all four lengths.
#
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --array=0-2
#SBATCH --time=24:00:00
#SBATCH --job-name=lclm-combo-eval
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-combo-eval-%A_%a.log

set -euo pipefail

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly python_bin="${PYTHON_BIN:-/data/home/scxj745/.conda/envs/qwen35/bin/python}"
readonly checkpoint="${CHECKPOINT:-${root}/artifacts/models/0.6b-4b-LCLM-8x}"
readonly task_index="${SLURM_ARRAY_TASK_ID:-0}"
readonly seeds=(6201 6202 6203)
readonly seed="${seeds[task_index]}"
readonly selector_checkpoint="${SELECTOR_CHECKPOINT:-${root}/artifacts/checkpoints/selector-combination-utility-roles-s${seed}}"
readonly result_stem="${RESULT_STEM:-combo-utility-roles-v2}"
readonly compute_answer_nll="${COMPUTE_ANSWER_NLL:-1}"

test -n "${seed:-}"
test -f "${checkpoint}/.download_complete"
test -f "${selector_checkpoint}/selector.safetensors"
test -f "${selector_checkpoint}/data_split.json"

cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

for tag in 4k 8k 16k 32k; do
  input_dir="${root}/artifacts/data/ruler_all_tasks_v2_${tag}"
  cache_dir="${root}/artifacts/runs/ruler-all-tasks-v2-${tag}-cache"
  work_dir="${root}/artifacts/runs/${result_stem}-25pct-ruler-all-${tag}-s${seed}"
  test -f "${input_dir}/oracle_latents.jsonl"
  test -f "${cache_dir}/manifest.json"
  eval_args=(
    --checkpoint "${checkpoint}"
    --input-dir "${input_dir}"
    --cache-dir "${cache_dir}"
    --work-dir "${work_dir}"
    --selection-ratio 0.25
    --selector-mode learned
    --selector-checkpoint "${selector_checkpoint}"
    --oracle-jsonl "${input_dir}/oracle_latents.jsonl"
    --position-policy full_layout
    --max-tokens 128
    --dtype bf16
    --random-seed "${seed}"
    --reuse-queries 1
    --repeat-each-query 1
    --max-resident-caches 1
  )
  if [[ "${compute_answer_nll}" = 1 ]]; then
    eval_args+=(--compute-answer-nll)
  fi
  "${python_bin}" -u -m inference.examples.eval_query_aware_niah "${eval_args[@]}"
done
