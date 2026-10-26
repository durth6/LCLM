#!/usr/bin/env bash
# Cache-reuse system measurement; repeated online requests are not new samples.
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=lclm-qa-reuse
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-qa-reuse-%j.log

set -euo pipefail

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly checkpoint="${root}/artifacts/models/0.6b-4b-LCLM-8x"
readonly input_dir="${root}/artifacts/data/ruler_niah_4k_smoke"
readonly cache_dir="${root}/artifacts/runs/smoke-k256/cache"
readonly reuse_queries="${REUSE_QUERIES:-4}"
readonly repeat_each_query="${REPEAT_EACH_QUERY:-4}"
readonly max_resident_caches="${MAX_RESIDENT_CACHES:-1}"
readonly result_suffix="${RESULT_SUFFIX:+-${RESULT_SUFFIX}}"
readonly work_dir="${root}/artifacts/runs/uniform-25pct-reuse4-g16-s4231${result_suffix}"

test -f "${checkpoint}/.download_complete"
test -f "${cache_dir}/manifest.json"
test -f "${input_dir}/oracle_latents.jsonl"

cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
exec /data/home/scxj745/.conda/envs/qwen35/bin/python -u \
  -m inference.examples.eval_query_aware_niah \
  --checkpoint "${checkpoint}" --input-dir "${input_dir}" --cache-dir "${cache_dir}" \
  --work-dir "${work_dir}" --selection-ratio 0.25 --selector-mode uniform \
  --oracle-jsonl "${input_dir}/oracle_latents.jsonl" \
  --repeat-each-query "${repeat_each_query}" --reuse-queries "${reuse_queries}" \
  --max-resident-caches "${max_resident_caches}" --max-tokens 16 --dtype bf16
