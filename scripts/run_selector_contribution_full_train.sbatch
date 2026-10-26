#!/usr/bin/env bash
# Merge all P1 contribution shards and train a full mixed-length selector.
# Intended to run after run_selector_contribution_full_array.sbatch.
#
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --job-name=lclm-p1-train
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-p1-train-%j.log

set -euo pipefail

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly checkpoint="${CHECKPOINT:-${root}/artifacts/models/0.6b-4b-LCLM-8x}"
readonly train_dir="${TRAIN_DIR:-${root}/artifacts/data/selector_train_mixed_4to32k}"
readonly shard_dir="${CONTRIBUTION_OUTPUT_DIR:-${train_dir}/contributions-zgrad-full}"
readonly contribution_jsonl="${CONTRIBUTION_JSONL:-${shard_dir}/contributions-zgrad-full.jsonl}"
readonly selector_checkpoint="${SELECTOR_CHECKPOINT:-${root}/artifacts/checkpoints/selector-stage-b-contribution-full-s6201}"
readonly seed="${SEED:-6201}"
readonly epochs="${EPOCHS:-5}"
readonly python_bin="${PYTHON_BIN:-/data/home/scxj745/.conda/envs/qwen35/bin/python}"

test -f "${checkpoint}/.download_complete"
test -f "${train_dir}/cache/manifest.json"
for shard in 0 1 2; do
  test -f "${shard_dir}/contributions-zgrad-shard-${shard}.jsonl"
done

cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"

# The 32K bucket is split into two 32-example jobs to fit a 4090.  Join those
# subshards here while the training GPU is already allocated.
if [[ ! -f "${shard_dir}/contributions-zgrad-shard-3.jsonl" ]]; then
  test -f "${shard_dir}/contributions-zgrad-shard-6.jsonl"
  test -f "${shard_dir}/contributions-zgrad-shard-7.jsonl"
  "${python_bin}" -u -m inference.examples.combine_contribution_sidecars \
    --inputs \
      "${shard_dir}/contributions-zgrad-shard-6.jsonl" \
      "${shard_dir}/contributions-zgrad-shard-7.jsonl" \
    --out "${shard_dir}/contributions-zgrad-shard-3.jsonl"
fi
for shard in 0 1 2 3; do
  test -f "${shard_dir}/contributions-zgrad-shard-${shard}.jsonl"
done

if [[ ! -f "${contribution_jsonl}" ]]; then
  "${python_bin}" -u -m inference.examples.combine_contribution_sidecars \
    --inputs \
      "${shard_dir}/contributions-zgrad-shard-0.jsonl" \
      "${shard_dir}/contributions-zgrad-shard-1.jsonl" \
      "${shard_dir}/contributions-zgrad-shard-2.jsonl" \
      "${shard_dir}/contributions-zgrad-shard-3.jsonl" \
    --queries-jsonl "${train_dir}/queries.jsonl" \
    --out "${contribution_jsonl}"
fi

exec "${python_bin}" -u -m train.train_selector \
  --checkpoint "${checkpoint}" \
  --cache-dir "${train_dir}/cache" \
  --queries-jsonl "${train_dir}/queries.jsonl" \
  --oracle-jsonl "${train_dir}/oracle_latents.jsonl" \
  --contribution-jsonl "${contribution_jsonl}" \
  --out "${selector_checkpoint}" \
  --epochs "${epochs}" \
  --learning-rate 3e-4 \
  --rank-weight 1.0 \
  --bce-weight 1.0 \
  --contribution-weight "${CONTRIBUTION_WEIGHT:-1.0}" \
  --contribution-temperature "${CONTRIBUTION_TEMPERATURE:-1.0}" \
  --contribution-normalization zgrad \
  --validation-fraction 0.125 \
  --eval-selection-ratio 0.25 \
  --seed "${seed}" \
  --dtype bf16 \
  --device cuda
