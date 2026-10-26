#!/usr/bin/env bash
# Compact-position control for the 4K all-task full-layout result.
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --job-name=lclm-pos-all
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-pos-all-%j.log

set -euo pipefail

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly python_bin=/data/home/scxj745/.conda/envs/qwen35/bin/python
readonly checkpoint=${root}/artifacts/models/0.6b-4b-LCLM-8x
readonly selector_checkpoint=${root}/artifacts/checkpoints/selector-p1-docsplit-posfix-s6201
readonly data_tag=${RULER_DATA_TAG:-}
readonly data_stem="ruler_all_tasks${data_tag:+_${data_tag}}"
readonly cache_stem="ruler-all-tasks${data_tag:+-${data_tag}}"
readonly result_suffix=${RESULT_SUFFIX:-${data_tag:-v1}}
readonly input_dir=${root}/artifacts/data/${data_stem}_4k
readonly cache_dir=${root}/artifacts/runs/${cache_stem}-4k-cache

test -f "${selector_checkpoint}/selector.safetensors"
test -f "${input_dir}/oracle_latents.jsonl"
test -f "${cache_dir}/manifest.json"
cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

exec "${python_bin}" -u -m inference.examples.eval_query_aware_niah \
  --checkpoint "${checkpoint}" \
  --input-dir "${input_dir}" \
  --cache-dir "${cache_dir}" \
  --work-dir "${root}/artifacts/runs/p1-docsplit-compact-${result_suffix}-25pct-ruler-all-4k-s6201" \
  --selection-ratio 0.25 \
  --selector-mode learned \
  --selector-checkpoint "${selector_checkpoint}" \
  --oracle-jsonl "${input_dir}/oracle_latents.jsonl" \
  --position-policy compact \
  --compute-answer-nll \
  --max-tokens 128 \
  --dtype bf16 \
  --random-seed 6201 \
  --reuse-queries 1 \
  --repeat-each-query 1 \
  --max-resident-caches 1
