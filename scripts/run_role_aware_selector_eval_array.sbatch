#!/usr/bin/env bash
# Full500 fixed-25% evaluation for the explicit role-aware selector.
#
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --array=0-11
#SBATCH --time=24:00:00
#SBATCH --job-name=lclm-role-eval
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-role-eval-%A_%a.log

set -euo pipefail
readonly root=/data/run01/scxj745/context-exp/LCLM
readonly python_bin="${PYTHON_BIN:-/data/home/scxj745/.conda/envs/qwen35/bin/python}"
readonly checkpoint="${CHECKPOINT:-${root}/artifacts/models/0.6b-4b-LCLM-8x}"
readonly task_index="${SLURM_ARRAY_TASK_ID:-0}"
readonly seeds=(6201 6202 6203)
readonly tags=(4k 8k 16k 32k)
readonly seed_index=$((task_index / ${#tags[@]}))
readonly length_index=$((task_index % ${#tags[@]}))
readonly seed="${seeds[seed_index]}"
readonly tag="${tags[length_index]}"
readonly data_tag="${RULER_DATA_TAG:-full500-v2}"
readonly input_dir="${root}/artifacts/data/ruler_all_tasks_${data_tag}_${tag}"
readonly cache_dir="${root}/artifacts/runs/ruler-all-tasks-${data_tag}-${tag}-cache"
readonly selector_checkpoint="${SELECTOR_CHECKPOINT:-${root}/artifacts/checkpoints/selector-role-aware-s${seed}}"
readonly role_jsonl="${input_dir}/oracle_latents_roles.jsonl"
readonly work_dir="${root}/artifacts/runs/${RESULT_PREFIX:-full500-role-aware-25pct}-${tag}-s${seed}"

if (( task_index < 0 || task_index >= 12 )); then exit 2; fi
test -f "${checkpoint}/.download_complete"
test -f "${input_dir}/queries.jsonl"
test -f "${cache_dir}/manifest.json"
test -f "${role_jsonl}"
test -f "${selector_checkpoint}/role_selector.safetensors"
test ! -e "${work_dir}/answers.jsonl"

cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

exec "${python_bin}" -u -m inference.examples.eval_role_aware_niah \
  --checkpoint "${checkpoint}" \
  --input-dir "${input_dir}" \
  --cache-dir "${cache_dir}" \
  --work-dir "${work_dir}" \
  --selector-checkpoint "${selector_checkpoint}" \
  --role-jsonl "${role_jsonl}" \
  --selection-ratio 0.25 \
  --position-policy full_layout \
  --compute-answer-nll \
  --max-tokens 128 \
  --random-seed "${seed}" \
  --dtype bf16 \
  --device cuda \
  --reuse-queries 1 \
  --repeat-each-query 1 \
  --max-resident-caches 1
