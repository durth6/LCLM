#!/usr/bin/env bash
# Prepare/cache all 13 RULER task families at 4K/8K/16K/32K.
# LIMIT_PER_TASK=0 means all 500 examples per task (6,500 per length).
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --array=0-3
#SBATCH --time=24:00:00
#SBATCH --job-name=lclm-ruler-all
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-ruler-all-%A_%a.log

set -euo pipefail

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly python_bin=/data/home/scxj745/.conda/envs/qwen35/bin/python
readonly checkpoint=${root}/artifacts/models/0.6b-4b-LCLM-8x
readonly task_index=${SLURM_ARRAY_TASK_ID:-0}
readonly limit_per_task=${LIMIT_PER_TASK:-0}
readonly data_tag=${RULER_DATA_TAG:-}

case "${task_index}" in
  0) context_length=4096; tag=4k ;;
  1) context_length=8192; tag=8k ;;
  2) context_length=16384; tag=16k ;;
  3) context_length=32768; tag=32k ;;
  *) echo "unsupported array index ${task_index}" >&2; exit 2 ;;
esac

readonly data_stem="ruler_all_tasks${data_tag:+_${data_tag}}"
readonly cache_stem="ruler-all-tasks${data_tag:+-${data_tag}}"
readonly input_dir=${root}/artifacts/data/${data_stem}_${tag}
readonly cache_dir=${root}/artifacts/runs/${cache_stem}-${tag}-cache
cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT=https://hf-mirror.com
# The full RULER revision is already cached locally. Avoid node-side mirror
# HEAD retries; set HF_HUB_OFFLINE=0 explicitly only when a cache refresh is needed.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

"${python_bin}" -u -m inference.examples.prepare_ruler_niah_query_aware \
  --ctx "${context_length}" \
  --tasks all \
  --limit-per-task "${limit_per_task}" \
  --out-dir "${input_dir}"

"${python_bin}" -u -m inference.query_aware cache \
  --checkpoint "${checkpoint}" \
  --memory-jsonl "${input_dir}/memory.jsonl" \
  --cache-dir "${cache_dir}" \
  --dtype bf16

exec "${python_bin}" -u -m inference.examples.annotate_oracle_latents \
  --checkpoint "${checkpoint}" \
  --memory-jsonl "${input_dir}/memory.jsonl" \
  --queries-jsonl "${input_dir}/queries.jsonl" \
  --cache-dir "${cache_dir}" \
  --out "${input_dir}/oracle_latents.jsonl"
