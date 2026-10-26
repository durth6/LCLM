#!/usr/bin/env bash
# Train the explicit multi-head role-aware selector on the independent
# synthetic role sidecar.  No RULER source text is opened by this job.
#
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --array=0-2
#SBATCH --time=06:00:00
#SBATCH --job-name=lclm-role-selector
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-role-selector-%A_%a.log

set -euo pipefail
readonly root=/data/run01/scxj745/context-exp/LCLM
readonly python_bin="${PYTHON_BIN:-/data/home/scxj745/.conda/envs/qwen35/bin/python}"
readonly checkpoint="${CHECKPOINT:-${root}/artifacts/models/0.6b-4b-LCLM-8x}"
readonly train_dir="${TRAIN_DIR:-${root}/artifacts/data/selector_train_mixed_4to32k}"
readonly task_index="${SLURM_ARRAY_TASK_ID:-0}"
readonly seeds=(6201 6202 6203)
readonly seed="${seeds[task_index]}"
readonly output="${root}/artifacts/checkpoints/selector-role-aware-s${seed}"

test -n "${seed:-}"
test -f "${checkpoint}/.download_complete"
test -f "${train_dir}/cache/manifest.json"
test -f "${train_dir}/queries.jsonl"
test -f "${train_dir}/oracle_latents_roles.jsonl"
test ! -e "${output}/role_selector.safetensors"

cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

exec "${python_bin}" -u -m train.train_role_aware_selector \
  --checkpoint "${checkpoint}" \
  --cache-dir "${train_dir}/cache" \
  --queries-jsonl "${train_dir}/queries.jsonl" \
  --role-oracle-jsonl "${train_dir}/oracle_latents_roles.jsonl" \
  --out "${output}" \
  --epochs "${EPOCHS:-10}" \
  --learning-rate 3e-4 \
  --retrieval-dim 128 \
  --min-evidence-fraction 0.08 \
  --selection-ratio 0.25 \
  --validation-fraction 0.125 \
  --split-seed 6201 \
  --seed "${seed}" \
  --dtype bf16 \
  --device cuda
