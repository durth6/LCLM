#!/usr/bin/env bash
# Evaluate one selector seed across four lengths; array 0-2 covers all seeds.
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --array=0-2
#SBATCH --time=24:00:00
#SBATCH --job-name=lclm-p1-ruler3
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-p1-ruler3-%A_%a.log

set -euo pipefail

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly python_bin=/data/home/scxj745/.conda/envs/qwen35/bin/python
readonly checkpoint=${root}/artifacts/models/0.6b-4b-LCLM-8x
readonly seeds=(6201 6202 6203)
readonly task_index=${SLURM_ARRAY_TASK_ID:-0}
readonly seed=${seeds[task_index]}
readonly selector_checkpoint=${root}/artifacts/checkpoints/selector-p1-docsplit-posfix-s${seed}
readonly compute_answer_nll=${COMPUTE_ANSWER_NLL:-1}
readonly data_tag=${RULER_DATA_TAG:-}
readonly data_stem="ruler_all_tasks${data_tag:+_${data_tag}}"
readonly cache_stem="ruler-all-tasks${data_tag:+-${data_tag}}"
readonly result_suffix=${RESULT_SUFFIX:-${data_tag:-v1}}

test -n "${seed:-}"
test -f "${selector_checkpoint}/selector.safetensors"
test -f "${selector_checkpoint}/data_split.json"
cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

for tag in 4k 8k 16k 32k; do
  input_dir=${root}/artifacts/data/${data_stem}_${tag}
  cache_dir=${root}/artifacts/runs/${cache_stem}-${tag}-cache
  test -f "${input_dir}/oracle_latents.jsonl"
  test -f "${cache_dir}/manifest.json"
  eval_args=(
    --checkpoint "${checkpoint}"
    --input-dir "${input_dir}"
    --cache-dir "${cache_dir}"
    --work-dir "${root}/artifacts/runs/p1-docsplit-full-layout-${result_suffix}-25pct-ruler-all-${tag}-s${seed}"
    --selection-ratio 0.25
    --selector-mode learned
    --selector-checkpoint "${selector_checkpoint}"
    --oracle-jsonl "${input_dir}/oracle_latents.jsonl"
    --position-policy full_layout
    --max-tokens 128
    --dtype bf16
    --random-seed "${seed}"
    --reuse-queries 1
    --repeat-each-query 1
    --max-resident-caches 1
  )
  if [[ "${compute_answer_nll}" = 1 ]]; then
    eval_args+=(--compute-answer-nll)
  fi
  "${python_bin}" -u -m inference.examples.eval_query_aware_niah "${eval_args[@]}"
done
