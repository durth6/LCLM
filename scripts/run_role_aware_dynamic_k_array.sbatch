#!/usr/bin/env bash
# Per-query confidence dynamic-K with RoleAwareSelector.  Budget decisions use
# only the query-conditioned relevance score mass; role labels are diagnostics.
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --array=0-11
#SBATCH --time=24:00:00
#SBATCH --job-name=lclm-role-dynamic
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-role-dynamic-%A_%a.log

set -euo pipefail
readonly root=/data/run01/scxj745/context-exp/LCLM
readonly python_bin="${PYTHON_BIN:-/data/home/scxj745/.conda/envs/qwen35/bin/python}"
readonly checkpoint="${CHECKPOINT:-${root}/artifacts/models/0.6b-4b-LCLM-8x}"
readonly index="${SLURM_ARRAY_TASK_ID:-0}"
readonly seeds=(6201 6202 6203)
readonly tags=(4k 8k 16k 32k)
readonly seed="${seeds[index / 4]}"
readonly tag="${tags[index % 4]}"
readonly data_tag="${RULER_DATA_TAG:-full500-v2}"
readonly input_dir="${root}/artifacts/data/ruler_all_tasks_${data_tag}_${tag}"
readonly cache_dir="${root}/artifacts/runs/ruler-all-tasks-${data_tag}-${tag}-cache"
readonly selector_dir="${root}/artifacts/checkpoints/selector-role-aware-s${seed}"
readonly calibration="${CALIBRATION_ROOT:-${root}/artifacts/calibration/role-dynamic-k}/calibration-s${seed}.json"
readonly work_dir="${root}/artifacts/runs/${RESULT_PREFIX:-full500-role-aware-calibrated-dynamic-k}-${tag}-s${seed}"

if (( index < 0 || index >= 12 )); then exit 2; fi
test -f "${checkpoint}/.download_complete"
test -f "${input_dir}/queries.jsonl"
test -f "${input_dir}/oracle_latents_roles.jsonl"
test -f "${cache_dir}/manifest.json"
test -f "${selector_dir}/role_selector.safetensors"
test -f "${calibration}"
test ! -e "${work_dir}/answers.jsonl"
test ! -e "${work_dir}/summary.json"

cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

exec "${python_bin}" -u -m inference.examples.eval_role_aware_dynamic_k \
  --checkpoint "${checkpoint}" \
  --cache-dir "${cache_dir}" \
  --queries-jsonl "${input_dir}/queries.jsonl" \
  --role-jsonl "${input_dir}/oracle_latents_roles.jsonl" \
  --selector-checkpoint "${selector_dir}" \
  --calibration "${calibration}" \
  --out "${work_dir}/answers.jsonl" \
  --selection-ratio 0.25 \
  --dynamic-budget-ratios 0.25,0.5,0.75,1.0 \
  --position-policy full_layout \
  --compute-answer-nll \
  --max-tokens 128 \
  --random-seed "${seed}" \
  --dtype bf16 \
  --device cuda
