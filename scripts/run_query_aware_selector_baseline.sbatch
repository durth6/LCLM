#!/usr/bin/env bash
# Examples:
#   SELECTOR_MODE=uniform SELECTION_RATIO=0.25 sbatch scripts/run_query_aware_selector_baseline.sbatch
#   SELECTOR_MODE=random SELECTION_RATIO=0.50 RANDOM_SEED=5201 sbatch scripts/run_query_aware_selector_baseline.sbatch
#   SELECTOR_MODE=learned SELECTION_RATIO=0.25 COVERAGE_RERANK=1 sbatch scripts/run_query_aware_selector_baseline.sbatch
#   SELECTOR_MODE=learned SELECTION_RATIO=0.25 DYNAMIC_K=1 sbatch scripts/run_query_aware_selector_baseline.sbatch
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --job-name=lclm-qa-baseline
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-qa-baseline-%j.log

set -euo pipefail
# Use the same REPEAT_EACH_QUERY/REUSE_QUERIES/MAX_RESIDENT_CACHES values as
# the full baseline.  Set RESULT_SUFFIX (for example, cold or resident) to
# keep protocol variants in separate result directories.

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly checkpoint="${root}/artifacts/models/0.6b-4b-LCLM-8x"
readonly input_dir="${NIAH_INPUT_DIR:-${root}/artifacts/data/ruler_niah_4k_smoke}"
readonly cache_dir="${CACHE_DIR:-${root}/artifacts/runs/smoke-k256/cache}"
readonly oracle_jsonl="${ORACLE_JSONL:-${input_dir}/oracle_latents.jsonl}"
readonly selector_mode="${SELECTOR_MODE:?set SELECTOR_MODE to cosine/learned/random/uniform/oracle}"
readonly selector_checkpoint="${SELECTOR_CHECKPOINT:-}"
readonly selection_ratio="${SELECTION_RATIO:?set SELECTION_RATIO in (0,1]}"
readonly random_seed="${RANDOM_SEED:-4231}"
readonly max_tokens="${MAX_TOKENS:-16}"
readonly neighbor_radius="${NEIGHBOR_RADIUS:-0}"
readonly coverage_rerank="${COVERAGE_RERANK:-0}"
readonly coverage_weight="${COVERAGE_WEIGHT:-0.25}"
readonly dynamic_k="${DYNAMIC_K:-0}"
readonly dynamic_coverage_threshold="${DYNAMIC_COVERAGE_THRESHOLD:-0.75}"
readonly dynamic_budget_ratios="${DYNAMIC_BUDGET_RATIOS:-0.25,0.5,0.75,1.0}"
readonly reuse_queries="${REUSE_QUERIES:-1}"
readonly repeat_each_query="${REPEAT_EACH_QUERY:-1}"
readonly max_resident_caches="${MAX_RESIDENT_CACHES:-1}"
readonly result_suffix="${RESULT_SUFFIX:+-${RESULT_SUFFIX}}"

# A filesystem-safe, stable tag (e.g. 0.25 -> 25pct).
case "${selection_ratio}" in
  0.125) tag=12p5pct ;;
  0.25) tag=25pct ;;
  0.5) tag=50pct ;;
  0.75) tag=75pct ;;
  1|1.0) tag=100pct ;;
  *) tag="ratio-${selection_ratio}" ;;
esac
# Keep decoding-budget variants separate: generated answer length is a material
# evaluation condition, not merely a logging preference.
policy_tag="relevance"
if [[ "${coverage_rerank}" = 1 ]]; then
  policy_tag="coverage"
fi
if [[ "${dynamic_k}" = 1 ]]; then
  policy_tag="dynamic"
fi
readonly work_dir="${root}/artifacts/runs/${selector_mode}-${policy_tag}-${tag}-n${neighbor_radius}-g${max_tokens}-s${random_seed}${result_suffix}"

test -f "${checkpoint}/.download_complete"
test -f "${cache_dir}/manifest.json"
test -f "${oracle_jsonl}"
if [[ "${selector_mode}" = learned ]]; then
  test -n "${selector_checkpoint}"
  test -f "${selector_checkpoint}/selector.safetensors"
elif [[ -n "${selector_checkpoint}" ]]; then
  echo "SELECTOR_CHECKPOINT is only valid with SELECTOR_MODE=learned" >&2
  exit 2
fi

selector_checkpoint_args=()
if [[ -n "${selector_checkpoint}" ]]; then
  selector_checkpoint_args+=(--selector-checkpoint "${selector_checkpoint}")
fi

coverage_args=(--coverage-weight "${coverage_weight}")
if [[ "${coverage_rerank}" = 1 ]]; then
  coverage_args+=(--coverage-rerank)
fi
dynamic_args=()
if [[ "${dynamic_k}" = 1 ]]; then
  dynamic_args+=(
    --dynamic-k
    --dynamic-coverage-threshold "${dynamic_coverage_threshold}"
    --dynamic-budget-ratios "${dynamic_budget_ratios}"
  )
fi

cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
exec /data/home/scxj745/.conda/envs/qwen35/bin/python -u \
  -m inference.examples.eval_query_aware_niah \
  --checkpoint "${checkpoint}" \
  --input-dir "${input_dir}" \
  --cache-dir "${cache_dir}" \
  --work-dir "${work_dir}" \
  --selection-ratio "${selection_ratio}" \
  --selector-mode "${selector_mode}" \
  "${selector_checkpoint_args[@]}" \
  --oracle-jsonl "${oracle_jsonl}" \
  --random-seed "${random_seed}" \
  --neighbor-radius "${neighbor_radius}" \
  --max-tokens "${max_tokens}" \
  --dtype bf16 \
  --reuse-queries "${reuse_queries}" \
  --repeat-each-query "${repeat_each_query}" \
  --max-resident-caches "${max_resident_caches}" \
  "${coverage_args[@]}" \
  "${dynamic_args[@]}"
