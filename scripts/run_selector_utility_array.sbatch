#!/usr/bin/env bash
# Annotate finite candidate-family answer-NLL utility targets for the
# synthetic selector training set.  Each array task owns a disjoint query
# slice and writes a provenance-rich v2 utility sidecar shard.
#
# Submit on one 4090 per shard, for example:
#   sbatch --array=0-7 --gpus=1 -p gpu_4090 scripts/run_selector_utility_array.sbatch
#
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --array=0-7
#SBATCH --time=08:00:00
#SBATCH --job-name=lclm-utility-label
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-utility-label-%A_%a.log

set -euo pipefail

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly python_bin="${PYTHON_BIN:-/data/home/scxj745/.conda/envs/qwen35/bin/python}"
readonly checkpoint="${CHECKPOINT:-${root}/artifacts/models/0.6b-4b-LCLM-8x}"
readonly train_dir="${TRAIN_DIR:-${root}/artifacts/data/selector_train_mixed_4to32k}"
readonly output_dir="${UTILITY_OUTPUT_DIR:-${train_dir}/utility-oracle-full-layout-v2}"
readonly shard_size="${SHARD_SIZE:-32}"
readonly shard_index="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
readonly start_index="${START_INDEX:-$((shard_index * shard_size))}"
readonly max_examples="${MAX_EXAMPLES:-${shard_size}}"
readonly target_tokens="${MAX_TARGET_TOKENS:-32}"
readonly num_groups="${NUM_GROUPS:-8}"
readonly neighbor_radius="${SPAN_NEIGHBOR_RADIUS:-1}"
readonly output="${OUTPUT:-${output_dir}/utility-oracle-shard-${shard_index}.jsonl}"

test "${shard_index}" -ge 0
test -f "${checkpoint}/.download_complete"
test -f "${train_dir}/cache/manifest.json"
test -f "${train_dir}/queries.jsonl"
test -f "${train_dir}/oracle_latents_roles.jsonl"
test -f "${train_dir}/contributions-zgrad-full/contributions-zgrad-full.jsonl"
mkdir -p "${output_dir}"

cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -f "${output}" ]] && [[ "$(wc -l < "${output}")" -eq "${max_examples}" ]]; then
  echo "reusing complete utility shard ${output}"
  exit 0
fi

exec "${python_bin}" -u -m inference.examples.annotate_utility_oracle \
  --checkpoint "${checkpoint}" \
  --cache-dir "${train_dir}/cache" \
  --queries-jsonl "${train_dir}/queries.jsonl" \
  --out "${output}" \
  --start-index "${start_index}" \
  --max-examples "${max_examples}" \
  --max-target-tokens "${target_tokens}" \
  --selection-ratio 0.25 \
  --num-groups "${num_groups}" \
  --span-neighbor-radius "${neighbor_radius}" \
  --span-oracle-jsonl "${train_dir}/oracle_latents_roles.jsonl" \
  --contribution-jsonl "${train_dir}/contributions-zgrad-full/contributions-zgrad-full.jsonl" \
  --position-policy full_layout \
  --device cuda \
  --dtype bf16
