#!/usr/bin/env bash
# Calibrate one role-aware selector seed on independent validation documents.
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --array=0-2
#SBATCH --time=04:00:00
#SBATCH --job-name=lclm-dynk-cal
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-dynk-cal-%A_%a.log

set -euo pipefail
readonly root=/data/run01/scxj745/context-exp/LCLM
readonly python_bin="${PYTHON_BIN:-/data/home/scxj745/.conda/envs/qwen35/bin/python}"
readonly checkpoint="${CHECKPOINT:-${root}/artifacts/models/0.6b-4b-LCLM-8x}"
readonly index="${SLURM_ARRAY_TASK_ID:-0}"
readonly seeds=(6201 6202 6203)
if (( index < 0 || index >= 3 )); then exit 2; fi
readonly seed="${seeds[index]}"
readonly data_dir="${DATA_DIR:-${root}/artifacts/data/dynamic_k_validation_v1}"
readonly cache_dir="${CACHE_DIR:-${root}/artifacts/runs/dynamic-k-validation-v1-cache}"
readonly selector_dir="${root}/artifacts/checkpoints/selector-role-aware-s${seed}"
readonly output_dir="${CALIBRATION_ROOT:-${root}/artifacts/calibration/role-dynamic-k}"
readonly output="${output_dir}/calibration-s${seed}.json"

test -f "${checkpoint}/.download_complete"
test -f "${data_dir}/queries.jsonl"
test -f "${data_dir}/oracle_latents_roles.jsonl"
test -f "${cache_dir}/manifest.json"
test -f "${selector_dir}/role_selector.safetensors"
test ! -e "${output}"
mkdir -p "${output_dir}"
cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

exec "${python_bin}" -u -m inference.examples.calibrate_role_dynamic_k \
  --checkpoint "${checkpoint}" \
  --cache-dir "${cache_dir}" \
  --queries-jsonl "${data_dir}/queries.jsonl" \
  --role-jsonl "${data_dir}/oracle_latents_roles.jsonl" \
  --selector-checkpoint "${selector_dir}" \
  --out "${output}" \
  --budget-ratios 0.25,0.5,0.75,1.0 \
  --temperatures "${CALIBRATION_TEMPERATURES:-0.02,0.05,0.1,0.2,0.5,1.0}" \
  --metric-tolerance "${METRIC_TOLERANCE:-0.005}" \
  --dtype bf16 \
  --device cuda
