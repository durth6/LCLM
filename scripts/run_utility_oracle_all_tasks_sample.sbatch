#!/usr/bin/env bash
# Direct answer-NLL oracle on one 4K example from each of all 13 RULER tasks.
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --job-name=lclm-util-all
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-util-all-%j.log

set -euo pipefail

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly python_bin=/data/home/scxj745/.conda/envs/qwen35/bin/python
readonly checkpoint=${root}/artifacts/models/0.6b-4b-LCLM-8x
readonly data_tag=${RULER_DATA_TAG:-}
readonly data_stem="ruler_all_tasks${data_tag:+_${data_tag}}"
readonly cache_stem="ruler-all-tasks${data_tag:+-${data_tag}}"
readonly result_suffix=${RESULT_SUFFIX:-${data_tag:-v1}}
readonly input_dir=${root}/artifacts/data/${data_stem}_4k
readonly cache_dir=${root}/artifacts/runs/${cache_stem}-4k-cache
readonly learned_answers=${root}/artifacts/runs/p1-docsplit-full-layout-${result_suffix}-25pct-ruler-all-4k-s6201/answers.jsonl
readonly output=${input_dir}/utility-oracle-group8-full-layout-${result_suffix}-1per-task.jsonl

test -f "${input_dir}/oracle_latents.jsonl"
test -f "${cache_dir}/manifest.json"
test -f "${learned_answers}"
cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "${python_bin}" -u -m inference.examples.annotate_utility_oracle \
  --checkpoint "${checkpoint}" \
  --cache-dir "${cache_dir}" \
  --queries-jsonl "${input_dir}/queries.jsonl" \
  --out "${output}" \
  --selection-ratio 0.25 \
  --num-groups 8 \
  --span-oracle-jsonl "${input_dir}/oracle_latents.jsonl" \
  --candidate-jsonl "learned_s6201=${learned_answers}" \
  --position-policy full_layout \
  --max-per-task 1 \
  --max-target-tokens 32 \
  --dtype bf16
