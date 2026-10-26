#!/usr/bin/env bash
# Step-matched control for the 4K--32K mixed-length Stage-A run.
# It trains only the *same* mix4k examples, cached in the mixed cache, with
# identical initialization/split seed.  20 epochs x 56 examples = 1120 updates,
# matching 5 epochs x 224 examples in the mixed run.
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=lclm-selector-4k-control
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-selector-4k-control-%j.log

set -euo pipefail

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly checkpoint="${root}/artifacts/models/0.6b-4b-LCLM-8x"
readonly train_dir="${root}/artifacts/data/selector_train_mix_4k"
readonly mixed_dir="${root}/artifacts/data/selector_train_mixed_4to32k"
readonly train_cache="${mixed_dir}/cache"
readonly train_oracle="${mixed_dir}/oracle_latents.jsonl"
readonly seed="${SEED:-6201}"
readonly epochs="${EPOCHS:-20}"
readonly selector_checkpoint="${root}/artifacts/checkpoints/selector-stage-a-mix4k-stepmatched-s${seed}"
readonly python=/data/home/scxj745/.conda/envs/qwen35/bin/python
readonly reuse_queries="${REUSE_QUERIES:-1}"
readonly repeat_each_query="${REPEAT_EACH_QUERY:-1}"
readonly max_resident_caches="${MAX_RESIDENT_CACHES:-1}"
readonly result_suffix="${RESULT_SUFFIX:+-${RESULT_SUFFIX}}"

test -f "${checkpoint}/.download_complete"
test -f "${train_dir}/queries.jsonl"
test -f "${train_oracle}"
test -f "${train_cache}/mix4k-000000/memory.safetensors"
cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"

"${python}" -u -m train.train_selector \
  --checkpoint "${checkpoint}" --cache-dir "${train_cache}" \
  --queries-jsonl "${train_dir}/queries.jsonl" --oracle-jsonl "${train_oracle}" \
  --out "${selector_checkpoint}" --epochs "${epochs}" --learning-rate 3e-4 \
  --validation-fraction 0.125 --eval-selection-ratio 0.25 --seed "${seed}" --dtype bf16

for tag in 4k 8k 16k 32k; do
  if [[ "${tag}" = 4k ]]; then
    input_dir="${root}/artifacts/data/ruler_niah_4k_smoke"
    cache_dir="${root}/artifacts/runs/smoke-k256/cache"
  else
    input_dir="${root}/artifacts/data/ruler_niah_${tag}_smoke"
    cache_dir="${root}/artifacts/runs/ruler-${tag}-cache"
  fi
  "${python}" -u -m inference.examples.eval_query_aware_niah \
    --checkpoint "${checkpoint}" --input-dir "${input_dir}" --cache-dir "${cache_dir}" \
    --work-dir "${root}/artifacts/runs/learned-25pct-n0-g128-ctx${tag}-4kcontrol-s${seed}${result_suffix}" \
    --selection-ratio 0.25 --selector-mode learned --selector-checkpoint "${selector_checkpoint}" \
    --oracle-jsonl "${input_dir}/oracle_latents.jsonl" --max-tokens 128 --dtype bf16 \
    --random-seed "${seed}" --reuse-queries "${reuse_queries}" \
    --repeat-each-query "${repeat_each_query}" --max-resident-caches "${max_resident_caches}"
done
