#!/usr/bin/env bash
# Offline role-label oracle-K curve, one seed×length per GPU task.
#
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --array=0-11
#SBATCH --time=08:00:00
#SBATCH --job-name=lclm-oracle-k
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-oracle-k-%A_%a.log

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
readonly output="${root}/artifacts/runs/${RESULT_PREFIX:-role-oracle-k-full500}-${tag}-s${seed}.json"

if (( task_index < 0 || task_index >= 12 )); then exit 2; fi
test -f "${checkpoint}/.download_complete"
test -f "${input_dir}/queries.jsonl"
test -f "${input_dir}/oracle_latents_roles.jsonl"
test -f "${cache_dir}/manifest.json"
test -f "${selector_checkpoint}/role_selector.safetensors"
test ! -e "${output}"

cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

exec "${python_bin}" -u -m inference.examples.analyze_role_oracle_k \
  --checkpoint "${checkpoint}" \
  --cache-dir "${cache_dir}" \
  --queries-jsonl "${input_dir}/queries.jsonl" \
  --role-jsonl "${input_dir}/oracle_latents_roles.jsonl" \
  --selector-checkpoint "${selector_checkpoint}" \
  --out "${output}" \
  --budget-ratios 0.25,0.5,0.75,1.0 \
  --joint-threshold 1.0 \
  --dtype bf16 \
  --device cuda
