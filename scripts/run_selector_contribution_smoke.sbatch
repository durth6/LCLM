#!/usr/bin/env bash
# P1 contribution-distillation smoke experiment.
#
# Default settings annotate and train on the first eight independent training
# examples, then evaluate at a fixed 25% latent budget on the 4K RULER smoke
# split.  Set CONTRIBUTION_MAX_EXAMPLES=0 for the full mixed-length sidecar.
#
# Example:
#   sbatch scripts/run_selector_contribution_smoke.sbatch
#   CONTRIBUTION_MAX_EXAMPLES=0 EPOCHS=5 sbatch scripts/run_selector_contribution_smoke.sbatch
#
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --job-name=lclm-selector-p1
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-selector-p1-%j.log

set -euo pipefail

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly checkpoint="${CHECKPOINT:-${root}/artifacts/models/0.6b-4b-LCLM-8x}"
readonly train_dir="${TRAIN_DIR:-${root}/artifacts/data/selector_train_mixed_4to32k}"
readonly train_cache="${train_dir}/cache"
readonly train_oracle="${train_dir}/oracle_latents.jsonl"
readonly max_examples="${CONTRIBUTION_MAX_EXAMPLES:-8}"
readonly target_tokens="${MAX_TARGET_TOKENS:-32}"
readonly contribution_weight="${CONTRIBUTION_WEIGHT:-1.0}"
readonly contribution_temperature="${CONTRIBUTION_TEMPERATURE:-1.0}"
readonly seed="${SEED:-6201}"
readonly epochs="${EPOCHS:-2}"
readonly max_train_examples="${TRAIN_MAX_EXAMPLES:-${max_examples}}"
readonly tag="${max_examples}examples"
readonly contribution_jsonl="${CONTRIBUTION_JSONL:-${train_dir}/contributions-zgrad-${tag}.jsonl}"
readonly selector_checkpoint="${SELECTOR_CHECKPOINT:-${root}/artifacts/checkpoints/selector-stage-b-contribution-${tag}-s${seed}}"
readonly eval_work_dir="${root}/artifacts/runs/p1-contribution-25pct-ctx4k-${tag}-s${seed}"
readonly python_bin="${PYTHON_BIN:-/data/home/scxj745/.conda/envs/qwen5090/bin/python}"

test -f "${checkpoint}/.download_complete"
test -f "${train_cache}/manifest.json"
test -f "${train_oracle}"
test -f "${root}/artifacts/data/ruler_niah_4k_smoke/oracle_latents.jsonl"
test -f "${root}/artifacts/runs/smoke-k256/cache/manifest.json"

cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -f "${contribution_jsonl}" ]]; then
  contribution_args=(
    --checkpoint "${checkpoint}"
    --cache-dir "${train_cache}"
    --queries-jsonl "${train_dir}/queries.jsonl"
    --out "${contribution_jsonl}"
    --max-examples "${max_examples}"
    --max-target-tokens "${target_tokens}"
    --device cuda
    --dtype bf16
  )
  if [[ "${LEAVE_ONE_OUT:-0}" = 1 ]]; then
    contribution_args+=(--leave-one-out --max-loo-latents "${MAX_LOO_LATENTS:-8}")
  fi
  "${python_bin}" -u -m inference.examples.annotate_contribution_latents "${contribution_args[@]}"
fi

train_args=(
  --checkpoint "${checkpoint}"
  --cache-dir "${train_cache}"
  --queries-jsonl "${train_dir}/queries.jsonl"
  --oracle-jsonl "${train_oracle}"
  --contribution-jsonl "${contribution_jsonl}"
  --out "${selector_checkpoint}"
  --max-examples "${max_train_examples}"
  --epochs "${epochs}"
  --learning-rate 3e-4
  --rank-weight 1.0
  --bce-weight 1.0
  --contribution-weight "${contribution_weight}"
  --contribution-temperature "${contribution_temperature}"
  --contribution-normalization zgrad
  --validation-fraction 0.25
  --eval-selection-ratio 0.25
  --seed "${seed}"
  --dtype bf16
  --device cuda
)
"${python_bin}" -u -m train.train_selector "${train_args[@]}"

"${python_bin}" -u -m inference.examples.eval_query_aware_niah \
  --checkpoint "${checkpoint}" \
  --input-dir "${root}/artifacts/data/ruler_niah_4k_smoke" \
  --cache-dir "${root}/artifacts/runs/smoke-k256/cache" \
  --work-dir "${eval_work_dir}" \
  --selection-ratio 0.25 \
  --selector-mode learned \
  --selector-checkpoint "${selector_checkpoint}" \
  --oracle-jsonl "${root}/artifacts/data/ruler_niah_4k_smoke/oracle_latents.jsonl" \
  --max-tokens 16 \
  --dtype bf16 \
  --random-seed "${seed}" \
  --reuse-queries 1 \
  --repeat-each-query 1 \
  --max-resident-caches 1
