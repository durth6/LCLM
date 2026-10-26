#!/usr/bin/env bash
# Strict original-LCLM baselines under the same end-to-end timing definition
# as the query-aware runs: disk latent-cache load is part of TTFT, and K=M.
# Existing pre-encoded RULER caches are reused; no raw context is read online.
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=lclm-qa-full
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-qa-full-%j.log

set -euo pipefail
# Controlled resident-cache comparison:
#   REPEAT_EACH_QUERY=2 REUSE_QUERIES=2 RESULT_SUFFIX=resident sbatch ...
# Keep the same values for full and learned runs; use RESULT_SUFFIX=cold for
# a separate output directory.  Cold-load values describe observed filesystem
# IO; resident TTFT is the primary cache-temperature-independent comparison.

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly checkpoint="${root}/artifacts/models/0.6b-4b-LCLM-8x"
readonly python=/data/home/scxj745/.conda/envs/qwen35/bin/python
readonly reuse_queries="${REUSE_QUERIES:-1}"
readonly repeat_each_query="${REPEAT_EACH_QUERY:-1}"
readonly max_resident_caches="${MAX_RESIDENT_CACHES:-1}"
readonly result_suffix="${RESULT_SUFFIX:+-${RESULT_SUFFIX}}"

test -f "${checkpoint}/.download_complete"
cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"

for tag in 4k 8k 16k 32k; do
  if [[ "${tag}" = 4k ]]; then
    input_dir="${root}/artifacts/data/ruler_niah_4k_smoke"
    cache_dir="${root}/artifacts/runs/smoke-k256/cache"
  else
    input_dir="${root}/artifacts/data/ruler_niah_${tag}_smoke"
    cache_dir="${root}/artifacts/runs/ruler-${tag}-cache"
  fi
  test -f "${input_dir}/queries.jsonl"
  test -f "${cache_dir}/manifest.json"

  "${python}" -u -m inference.examples.eval_query_aware_niah \
    --checkpoint "${checkpoint}" --input-dir "${input_dir}" --cache-dir "${cache_dir}" \
    --work-dir "${root}/artifacts/runs/full-100pct-n0-g128-ctx${tag}-strict-s6201${result_suffix}" \
    --selection-ratio 1.0 --selector-mode full --max-tokens 128 --dtype bf16 \
    --random-seed 6201 --reuse-queries "${reuse_queries}" \
    --repeat-each-query "${repeat_each_query}" \
    --max-resident-caches "${max_resident_caches}"
done
