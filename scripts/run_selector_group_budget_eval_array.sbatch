#!/usr/bin/env bash
# Evaluate the group-coverage selector at one fixed K/M budget on the corrected
# RULER matrix.  BUDGET_RATIO must be one of 0.25/0.5/0.75/1.0; each array
# index maps to one (seed, context length) pair.
#
# Example:
#   sbatch --array=0-11 \
#     --export=ALL,RULER_DATA_TAG=full500-v2,BUDGET_RATIO=0.5,RESULT_STEM=full500-group-fixed \
#     --gpus=1 -p gpu_4090 scripts/run_selector_group_budget_eval_array.sbatch
#
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --array=0-11
#SBATCH --time=24:00:00
#SBATCH --job-name=lclm-group-budget
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-group-budget-%A_%a.log

set -euo pipefail

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly python_bin="${PYTHON_BIN:-/data/home/scxj745/.conda/envs/qwen35/bin/python}"
readonly checkpoint="${CHECKPOINT:-${root}/artifacts/models/0.6b-4b-LCLM-8x}"
readonly task_index="${SLURM_ARRAY_TASK_ID:-0}"
readonly data_tag="${RULER_DATA_TAG:-v2}"
readonly budget_ratio="${BUDGET_RATIO:-0.5}"
readonly compute_answer_nll="${COMPUTE_ANSWER_NLL:-1}"
readonly seeds=(6201 6202 6203)
readonly tags=(4k 8k 16k 32k)

case "${budget_ratio}" in
  0.25) readonly budget_tag=25pct ;;
  0.5|0.50) readonly budget_tag=50pct ;;
  0.75) readonly budget_tag=75pct ;;
  1|1.0|1.00) readonly budget_tag=100pct ;;
  *)
    echo "unsupported BUDGET_RATIO=${budget_ratio}; expected 0.25, 0.5, 0.75, or 1.0" >&2
    exit 2
    ;;
esac

if (( task_index < 0 || task_index >= ${#seeds[@]} * ${#tags[@]} )); then
  echo "unsupported SLURM_ARRAY_TASK_ID=${task_index}" >&2
  exit 2
fi

readonly seed_index=$((task_index / ${#tags[@]}))
readonly length_index=$((task_index % ${#tags[@]}))
readonly seed="${seeds[seed_index]}"
readonly tag="${tags[length_index]}"
readonly selector_checkpoint="${SELECTOR_CHECKPOINT:-${root}/artifacts/checkpoints/selector-group-coverage-utility-s${seed}}"
readonly data_stem="ruler_all_tasks_${data_tag}"
readonly cache_stem="ruler-all-tasks-${data_tag}"
readonly input_dir="${root}/artifacts/data/${data_stem}_${tag}"
readonly cache_dir="${root}/artifacts/runs/${cache_stem}-${tag}-cache"
readonly result_stem="${RESULT_STEM:-group-coverage-fixed}"
readonly work_dir="${WORK_DIR:-${root}/artifacts/runs/${result_stem}-${budget_tag}-${tag}-s${seed}}"

if [[ -n "${ROLE_ORACLE_JSONL:-}" ]]; then
  readonly diagnostic_oracle_jsonl="${ROLE_ORACLE_JSONL}"
elif [[ -f "${input_dir}/oracle_latents_roles.jsonl" ]]; then
  readonly diagnostic_oracle_jsonl="${input_dir}/oracle_latents_roles.jsonl"
else
  readonly diagnostic_oracle_jsonl="${input_dir}/oracle_latents.jsonl"
fi

test -f "${checkpoint}/.download_complete"
test -f "${selector_checkpoint}/selector.safetensors"
test -f "${selector_checkpoint}/data_split.json"
test -f "${input_dir}/queries.jsonl"
test -f "${input_dir}/oracle_latents.jsonl"
test -f "${diagnostic_oracle_jsonl}"
test -f "${cache_dir}/manifest.json"
test ! -e "${work_dir}/answers.jsonl"
test ! -e "${work_dir}/summary.json"

cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

eval_args=(
  --checkpoint "${checkpoint}"
  --input-dir "${input_dir}"
  --cache-dir "${cache_dir}"
  --work-dir "${work_dir}"
  --selection-ratio "${budget_ratio}"
  --selector-mode learned
  --selector-checkpoint "${selector_checkpoint}"
  --oracle-jsonl "${diagnostic_oracle_jsonl}"
  --role-jsonl "${diagnostic_oracle_jsonl}"
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

exec "${python_bin}" -u -m inference.examples.eval_query_aware_niah "${eval_args[@]}"
