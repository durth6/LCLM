#!/usr/bin/env bash
# Three-seed fixed-25% Full500 evaluation for RoleAwareSelector.
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
readonly index="${SLURM_ARRAY_TASK_ID:-0}"
readonly seeds=(6201 6202 6203)
readonly tags=(4k 8k 16k 32k)
readonly seed="${seeds[index / 4]}"
readonly tag="${tags[index % 4]}"
readonly data_tag="${RULER_DATA_TAG:-full500-v2}"
readonly input_dir="${root}/artifacts/data/ruler_all_tasks_${data_tag}_${tag}"
readonly cache_dir="${root}/artifacts/runs/ruler-all-tasks-${data_tag}-${tag}-cache"
readonly selector_dir="${root}/artifacts/checkpoints/selector-role-aware-s${seed}"
readonly selection_ratio="${SELECTION_RATIO:-0.25}"
readonly work_dir="${root}/artifacts/runs/${RESULT_PREFIX:-full500-role-aware-25pct}-${tag}-s${seed}"

if (( index < 0 || index >= 12 )); then exit 2; fi
test -f "${checkpoint}/.download_complete"
test -f "${input_dir}/queries.jsonl"
test -f "${input_dir}/oracle_latents_roles.jsonl"
test -f "${cache_dir}/manifest.json"
test -f "${selector_dir}/role_selector.safetensors"
test ! -e "${work_dir}/answers.jsonl"

cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

exec "${python_bin}" -u -m inference.examples.eval_role_aware_full500 \
  --checkpoint "${checkpoint}" \
  --input-dir "${input_dir}" \
  --cache-dir "${cache_dir}" \
  --work-dir "${work_dir}" \
  --selector-checkpoint "${selector_dir}" \
  --role-jsonl "${input_dir}/oracle_latents_roles.jsonl" \
  --selection-ratio "${selection_ratio}" \
  --position-policy full_layout \
  --compute-answer-nll \
  --max-tokens 128 \
  --random-seed "${seed}" \
  --dtype bf16 \
  --device cuda
