#!/usr/bin/env bash
# Validate gradient contribution targets against real leave-one-out NLL deltas.
# This deliberately uses one 4K training context and a small ablation subset.
#
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --job-name=lclm-contrib-loo
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-contrib-loo-%j.log

set -euo pipefail

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly checkpoint="${CHECKPOINT:-${root}/artifacts/models/0.6b-4b-LCLM-8x}"
readonly train_dir="${TRAIN_DIR:-${root}/artifacts/data/selector_train_mixed_4to32k}"
readonly output="${OUTPUT:-${train_dir}/contributions-zgrad-loo-full-layout-v2-1example.jsonl}"
readonly max_loo_latents="${MAX_LOO_LATENTS:-8}"
readonly python_bin="${PYTHON_BIN:-/data/home/scxj745/.conda/envs/qwen5090/bin/python}"
readonly position_policy="${POSITION_POLICY:-full_layout}"

test -f "${checkpoint}/.download_complete"
test -f "${train_dir}/cache/manifest.json"
cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"

"${python_bin}" -u -m inference.examples.annotate_contribution_latents \
  --checkpoint "${checkpoint}" \
  --cache-dir "${train_dir}/cache" \
  --queries-jsonl "${train_dir}/queries.jsonl" \
  --out "${output}" \
  --max-examples 1 \
  --max-target-tokens 32 \
  --leave-one-out \
  --max-loo-latents "${max_loo_latents}" \
  --position-policy "${position_policy}" \
  --device cuda \
  --dtype bf16
