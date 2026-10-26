#!/usr/bin/env bash
# Full 256-example contribution annotation, sharded by context length.
# The 32K bucket can be split into task IDs 6-7 with SHARD_SIZE=32 so two
# independent 4090 jobs process it concurrently.
#
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --job-name=lclm-p1-label
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-p1-label-%A_%a.log

set -euo pipefail

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly checkpoint="${CHECKPOINT:-${root}/artifacts/models/0.6b-4b-LCLM-8x}"
readonly train_dir="${TRAIN_DIR:-${root}/artifacts/data/selector_train_mixed_4to32k}"
readonly shard_size="${SHARD_SIZE:-64}"
readonly shard_index="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
readonly start_index="${START_INDEX:-$((shard_index * shard_size))}"
readonly max_examples="${MAX_EXAMPLES:-${shard_size}}"
readonly target_tokens="${MAX_TARGET_TOKENS:-32}"
readonly activation_checkpointing="${ACTIVATION_CHECKPOINTING:-1}"
readonly output_dir="${CONTRIBUTION_OUTPUT_DIR:-${train_dir}/contributions-zgrad-full}"
readonly output="${OUTPUT:-${output_dir}/contributions-zgrad-shard-${shard_index}.jsonl}"
readonly python_bin="${PYTHON_BIN:-/data/home/scxj745/.conda/envs/qwen35/bin/python}"

test "${shard_index}" -ge 0
test -f "${checkpoint}/.download_complete"
test -f "${train_dir}/cache/manifest.json"
mkdir -p "${output_dir}"
cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -f "${output}" ]] && [[ "$(wc -l < "${output}")" -eq "${shard_size}" ]]; then
  echo "reusing complete shard ${output}"
  exit 0
fi

annotation_args=(
  --checkpoint "${checkpoint}" \
  --cache-dir "${train_dir}/cache" \
  --queries-jsonl "${train_dir}/queries.jsonl" \
  --out "${output}" \
  --start-index "${start_index}" \
  --max-examples "${max_examples}" \
  --max-target-tokens "${target_tokens}" \
  --device cuda \
  --dtype bf16
)
if [[ "${activation_checkpointing}" = 1 ]]; then
  annotation_args+=(--activation-checkpointing)
fi
exec "${python_bin}" -u -m inference.examples.annotate_contribution_latents "${annotation_args[@]}"
