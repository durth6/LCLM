#!/usr/bin/env bash
# Independent-data Stage-A smoke followed by a held-out RULER evaluation.
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=lclm-selector-a-smoke
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-selector-a-smoke-%j.log

set -euo pipefail

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly checkpoint="${root}/artifacts/models/0.6b-4b-LCLM-8x"
readonly train_seed="${TRAIN_SEED:-5201}"
readonly train_dir="${TRAIN_DIR:-${root}/artifacts/data/selector_train_synthetic_4k_smoke}"
readonly train_cache="${train_dir}/cache"
readonly train_oracle="${train_dir}/oracle_latents.jsonl"
readonly selector_checkpoint="${SELECTOR_CHECKPOINT:-${root}/artifacts/checkpoints/selector-stage-a-synthetic-4k-smoke-s${train_seed}}"
readonly eval_dir="${root}/artifacts/data/ruler_niah_4k_smoke"
readonly eval_cache="${root}/artifacts/runs/smoke-k256/cache"
readonly eval_oracle="${eval_dir}/oracle_latents.jsonl"

test -f "${checkpoint}/.download_complete"
test -f "${train_dir}/memory.jsonl"
test -f "${train_dir}/queries.jsonl"
test -f "${eval_cache}/manifest.json"
test -f "${eval_oracle}"

cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
readonly python=/data/home/scxj745/.conda/envs/qwen35/bin/python

"${python}" -u -m inference.query_aware cache \
  --checkpoint "${checkpoint}" --memory-jsonl "${train_dir}/memory.jsonl" \
  --cache-dir "${train_cache}" --dtype bf16
"${python}" -u -m inference.examples.annotate_oracle_latents \
  --checkpoint "${checkpoint}" --memory-jsonl "${train_dir}/memory.jsonl" \
  --queries-jsonl "${train_dir}/queries.jsonl" --cache-dir "${train_cache}" \
  --out "${train_oracle}"
"${python}" -u -m train.train_selector \
  --checkpoint "${checkpoint}" --cache-dir "${train_cache}" \
  --queries-jsonl "${train_dir}/queries.jsonl" --oracle-jsonl "${train_oracle}" \
  --out "${selector_checkpoint}" --epochs 10 --learning-rate 3e-4 \
  --validation-fraction 0.125 --eval-selection-ratio 0.25 --seed "${train_seed}" --dtype bf16
for selection_ratio in 0.25 0.5; do
  tag="$(test "${selection_ratio}" = 0.25 && printf 25pct || printf 50pct)"
  "${python}" -u -m inference.examples.eval_query_aware_niah \
    --checkpoint "${checkpoint}" --input-dir "${eval_dir}" --cache-dir "${eval_cache}" \
    --work-dir "${root}/artifacts/runs/learned-${tag}-n0-g128-s${train_seed}" \
    --selection-ratio "${selection_ratio}" --selector-mode learned \
    --selector-checkpoint "${selector_checkpoint}" --oracle-jsonl "${eval_oracle}" \
    --max-tokens 128 --dtype bf16 --random-seed "${train_seed}" --reuse-queries 1
done
