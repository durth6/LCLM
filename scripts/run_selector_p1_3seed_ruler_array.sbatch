#!/usr/bin/env bash
# Evaluate 3 selector seeds x 4 lengths on all 13 RULER task families.
# Submit with afterok dependency on run_selector_p1_3seed_train_array.sbatch.
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --array=0-11
#SBATCH --time=24:00:00
#SBATCH --job-name=lclm-p1-ruler3
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-p1-ruler3-%A_%a.log

set -euo pipefail

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly python_bin=/data/home/scxj745/.conda/envs/qwen35/bin/python
readonly checkpoint=${root}/artifacts/models/0.6b-4b-LCLM-8x
readonly seeds=(6201 6202 6203)
readonly tags=(4k 8k 16k 32k)
readonly task_index=${SLURM_ARRAY_TASK_ID:-0}
readonly seed_index=$((task_index / 4))
readonly length_index=$((task_index % 4))
readonly seed=${seeds[seed_index]}
readonly tag=${tags[length_index]}
readonly selector_checkpoint=${root}/artifacts/checkpoints/selector-p1-docsplit-posfix-s${seed}
readonly data_tag=${RULER_DATA_TAG:-v2}
readonly result_stem=${RESULT_STEM:-p1-docsplit-full-layout-25pct-ruler-all}
readonly input_dir=${root}/artifacts/data/ruler_all_tasks_${data_tag}_${tag}
readonly cache_dir=${root}/artifacts/runs/ruler-all-tasks-${data_tag}-${tag}-cache
readonly compute_answer_nll=${COMPUTE_ANSWER_NLL:-1}

test -f "${selector_checkpoint}/selector.safetensors"
test -f "${selector_checkpoint}/data_split.json"
test -f "${input_dir}/oracle_latents.jsonl"
test -f "${cache_dir}/manifest.json"
cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

eval_args=(
  --checkpoint "${checkpoint}"
  --input-dir "${input_dir}"
  --cache-dir "${cache_dir}"
  --work-dir "${root}/artifacts/runs/${result_stem}-${tag}-s${seed}"
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
exec "${python_bin}" -u -m inference.examples.eval_query_aware_niah "${eval_args[@]}"
