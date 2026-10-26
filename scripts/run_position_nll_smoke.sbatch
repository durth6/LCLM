#!/usr/bin/env bash
# Compare compact versus full-layout RoPE and measure answer-NLL retention.
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --array=0-1
#SBATCH --time=02:00:00
#SBATCH --job-name=lclm-pos-nll
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-pos-nll-%A_%a.log

set -euo pipefail

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly python_bin=/data/home/scxj745/.conda/envs/qwen35/bin/python
readonly checkpoint=${root}/artifacts/models/0.6b-4b-LCLM-8x
readonly selector_checkpoint=${root}/artifacts/checkpoints/selector-stage-b-contribution-full-s6201
readonly input_dir=${root}/artifacts/data/ruler_niah_4k_smoke
readonly cache_dir=${root}/artifacts/runs/smoke-k256/cache
readonly task_index=${SLURM_ARRAY_TASK_ID:-0}
readonly max_examples=${MAX_EXAMPLES:-8}

case "${task_index}" in
  0) position_policy=compact ;;
  1) position_policy=full_layout ;;
  *) echo "unsupported array index ${task_index}" >&2; exit 2 ;;
esac

test -f "${selector_checkpoint}/selector.safetensors"
test -f "${cache_dir}/manifest.json"
cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT=https://hf-mirror.com

exec "${python_bin}" -u -m inference.examples.eval_query_aware_niah \
  --checkpoint "${checkpoint}" \
  --input-dir "${input_dir}" \
  --cache-dir "${cache_dir}" \
  --work-dir "${root}/artifacts/runs/position-${position_policy}-nll-4k-${max_examples}examples-s6201" \
  --selection-ratio 0.25 \
  --selector-mode learned \
  --selector-checkpoint "${selector_checkpoint}" \
  --oracle-jsonl "${input_dir}/oracle_latents.jsonl" \
  --position-policy "${position_policy}" \
  --compute-answer-nll \
  --max-examples "${max_examples}" \
  --max-tokens 32 \
  --dtype bf16 \
  --random-seed 6201 \
  --reuse-queries 1 \
  --repeat-each-query 1 \
  --max-resident-caches 1
