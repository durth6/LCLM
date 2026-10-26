#!/usr/bin/env bash
# Merge all utility-oracle shards after run_selector_utility_array.sbatch.
#
# Submit after the annotation array, for example:
#   sbatch --dependency=afterok:<array_job_id> --gpus=1 -p gpu_4090 scripts/run_selector_utility_merge.sbatch
#
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --job-name=lclm-utility-merge
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-utility-merge-%j.log

set -euo pipefail

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly python_bin="${PYTHON_BIN:-/data/home/scxj745/.conda/envs/qwen35/bin/python}"
readonly train_dir="${TRAIN_DIR:-${root}/artifacts/data/selector_train_mixed_4to32k}"
readonly shard_dir="${UTILITY_OUTPUT_DIR:-${train_dir}/utility-oracle-full-layout-v2}"
readonly output="${UTILITY_JSONL:-${shard_dir}/utility-oracle-full-layout-v2.jsonl}"

test -f "${train_dir}/queries.jsonl"
for shard_index in 0 1 2 3 4 5 6 7; do
  test -f "${shard_dir}/utility-oracle-shard-${shard_index}.jsonl"
done

cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

"${python_bin}" -u -m inference.examples.combine_utility_sidecars \
  --inputs \
    "${shard_dir}/utility-oracle-shard-0.jsonl" \
    "${shard_dir}/utility-oracle-shard-1.jsonl" \
    "${shard_dir}/utility-oracle-shard-2.jsonl" \
    "${shard_dir}/utility-oracle-shard-3.jsonl" \
    "${shard_dir}/utility-oracle-shard-4.jsonl" \
    "${shard_dir}/utility-oracle-shard-5.jsonl" \
    "${shard_dir}/utility-oracle-shard-6.jsonl" \
    "${shard_dir}/utility-oracle-shard-7.jsonl" \
  --queries-jsonl "${train_dir}/queries.jsonl" \
  --out "${output}"

test "$(wc -l < "${output}")" -eq "$(wc -l < "${train_dir}/queries.jsonl")"
echo "merged $(wc -l < "${output}") utility rows into ${output}"
