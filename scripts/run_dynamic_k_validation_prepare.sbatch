#!/usr/bin/env bash
# Build an independent hard multi-key validation cache and offline role labels.
# This data is used only to calibrate the dynamic-K stopping rule.
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --job-name=lclm-dynk-val
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-dynk-val-%j.log

set -euo pipefail
readonly root=/data/run01/scxj745/context-exp/LCLM
readonly python_bin="${PYTHON_BIN:-/data/home/scxj745/.conda/envs/qwen35/bin/python}"
readonly checkpoint="${CHECKPOINT:-${root}/artifacts/models/0.6b-4b-LCLM-8x}"
readonly data_dir="${DATA_DIR:-${root}/artifacts/data/dynamic_k_validation_v1}"
readonly cache_dir="${CACHE_DIR:-${root}/artifacts/runs/dynamic-k-validation-v1-cache}"

test -f "${checkpoint}/.download_complete"
test ! -e "${data_dir}/queries.jsonl"
test ! -e "${cache_dir}/manifest.json"
mkdir -p "${data_dir}"
cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

"${python_bin}" -m pytest -q \
  tests/test_joint_role_analysis.py \
  tests/test_role_dynamic_k.py \
  tests/test_synthetic_selector_data.py

"${python_bin}" -u -m inference.examples.prepare_dynamic_k_validation \
  --out-dir "${data_dir}" \
  --examples-per-length "${EXAMPLES_PER_LENGTH:-64}" \
  --pair-count "${PAIR_COUNT:-24}" \
  --asked-counts "${ASKED_COUNTS:-2,4,8,12,16}" \
  --seed 8301

"${python_bin}" -u -m inference.examples.annotate_synthetic_roles \
  --memory-jsonl "${data_dir}/memory.jsonl" \
  --queries-jsonl "${data_dir}/queries.jsonl" \
  --out "${data_dir}/role_spans.jsonl" \
  --support-radius "${SUPPORT_RADIUS:-96}"

"${python_bin}" -u -m inference.query_aware cache \
  --checkpoint "${checkpoint}" \
  --memory-jsonl "${data_dir}/memory.jsonl" \
  --cache-dir "${cache_dir}" \
  --dtype bf16 \
  --device cuda

exec "${python_bin}" -u -m inference.examples.annotate_oracle_latents \
  --checkpoint "${checkpoint}" \
  --memory-jsonl "${data_dir}/memory.jsonl" \
  --queries-jsonl "${data_dir}/queries.jsonl" \
  --cache-dir "${cache_dir}" \
  --role-spans-jsonl "${data_dir}/role_spans.jsonl" \
  --out "${data_dir}/oracle_latents_roles.jsonl"
