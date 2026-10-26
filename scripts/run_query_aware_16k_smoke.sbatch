#!/usr/bin/env bash
# Long-context validation: 16K/32K RULER NIAH, full LCLM versus learned 25%.
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=lclm-qa-16k
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-qa-16k-%j.log

set -euo pipefail

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly checkpoint="${root}/artifacts/models/0.6b-4b-LCLM-8x"
readonly context_length="${CONTEXT_LENGTH:-16384}"
case "${context_length}" in
  8192) context_tag=8k ;;
  16384) context_tag=16k ;;
  32768) context_tag=32k ;;
  *) echo "CONTEXT_LENGTH must be 8192, 16384, or 32768" >&2; exit 2 ;;
esac
readonly input_dir="${root}/artifacts/data/ruler_niah_${context_tag}_smoke"
readonly cache_dir="${root}/artifacts/runs/ruler-${context_tag}-cache"
readonly selector_checkpoint="${root}/artifacts/checkpoints/selector-stage-a-synthetic-4k-smoke-s5201"
readonly reuse_queries="${REUSE_QUERIES:-1}"
readonly repeat_each_query="${REPEAT_EACH_QUERY:-1}"
readonly max_resident_caches="${MAX_RESIDENT_CACHES:-1}"
readonly result_suffix="${RESULT_SUFFIX:+-${RESULT_SUFFIX}}"

test -f "${checkpoint}/.download_complete"
test -f "${selector_checkpoint}/selector.safetensors"
cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
readonly python=/data/home/scxj745/.conda/envs/qwen35/bin/python

if [[ ! -f "${input_dir}/memory.jsonl" ]]; then
  "${python}" -u -m inference.examples.prepare_ruler_niah_query_aware \
    --ctx "${context_length}" --out-dir "${input_dir}" --limit-per-task 8
fi
if [[ ! -f "${cache_dir}/manifest.json" ]]; then
  "${python}" -u -m inference.query_aware cache \
    --checkpoint "${checkpoint}" --memory-jsonl "${input_dir}/memory.jsonl" \
    --cache-dir "${cache_dir}" --dtype bf16
fi
if [[ ! -f "${input_dir}/oracle_latents.jsonl" ]]; then
  "${python}" -u -m inference.examples.annotate_oracle_latents \
    --checkpoint "${checkpoint}" --memory-jsonl "${input_dir}/memory.jsonl" \
    --queries-jsonl "${input_dir}/queries.jsonl" --cache-dir "${cache_dir}" \
    --out "${input_dir}/oracle_latents.jsonl"
fi
"${python}" -u -m inference.examples.eval_query_aware_niah \
  --checkpoint "${checkpoint}" --input-dir "${input_dir}" --cache-dir "${cache_dir}" \
  --work-dir "${root}/artifacts/runs/uniform-100pct-n0-g128-ctx${context_tag}-s4231${result_suffix}" \
  --selection-ratio 1.0 --selector-mode uniform --oracle-jsonl "${input_dir}/oracle_latents.jsonl" \
  --max-tokens 128 --dtype bf16 --reuse-queries "${reuse_queries}" \
  --repeat-each-query "${repeat_each_query}" \
  --max-resident-caches "${max_resident_caches}"
"${python}" -u -m inference.examples.eval_query_aware_niah \
  --checkpoint "${checkpoint}" --input-dir "${input_dir}" --cache-dir "${cache_dir}" \
  --work-dir "${root}/artifacts/runs/learned-25pct-n0-g128-ctx${context_tag}-s5201${result_suffix}" \
  --selection-ratio 0.25 --selector-mode learned --selector-checkpoint "${selector_checkpoint}" \
  --oracle-jsonl "${input_dir}/oracle_latents.jsonl" --max-tokens 128 --dtype bf16 \
  --reuse-queries "${reuse_queries}" --repeat-each-query "${repeat_each_query}" \
  --max-resident-caches "${max_resident_caches}"
