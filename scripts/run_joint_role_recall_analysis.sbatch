#!/usr/bin/env bash
# Post-hoc joint-role recall conditioning.  This is label-only analysis and
# never enters answer-time selection; it runs after the full-LCLM comparison.
#
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --job-name=lclm-joint-role
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-joint-role-%j.log

set -euo pipefail
readonly root=/data/run01/scxj745/context-exp/LCLM
readonly python_bin="${PYTHON_BIN:-/data/home/scxj745/.conda/envs/qwen35/bin/python}"
readonly data_root="${root}/artifacts/data"
readonly run_root="${root}/artifacts/runs"
readonly output_root="${OUTPUT_ROOT:-${run_root}/joint-role-recall-full500}"
mkdir -p "${output_root}"
cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

for tag in 8k 16k 32k; do
  role_jsonl="${data_root}/ruler_all_tasks_full500-v2_${tag}/oracle_latents_roles.jsonl"
  args=(
    --role-jsonl "${role_jsonl}"
    --out "${output_root}/joint-role-${tag}.json"
    --answers "group25=${run_root}/full500-group-relevance-v2-25pct-${tag}-s6201/answers.jsonl"
    --answers "fixed50=${run_root}/full500-group-fixed-50pct-${tag}-s6201/answers.jsonl"
    --answers "fixed75=${run_root}/full500-group-fixed-75pct-${tag}-s6201/answers.jsonl"
    --answers "fixed100=${run_root}/full500-group-fixed-100pct-${tag}-s6201/answers.jsonl"
    --answers "full=${run_root}/full500-full-lclm-niah-${tag}/answers.jsonl"
    --answers "role_oracle=${run_root}/ruler-role-oracle-full500-v2-key_answer_support-25pct-${tag}-s6202/answers.jsonl"
  )
  for value in "${args[@]}"; do
    if [[ "${value}" == --answers ]]; then continue; fi
  done
  for path in \
    "${run_root}/full500-group-relevance-v2-25pct-${tag}-s6201/answers.jsonl" \
    "${run_root}/full500-group-fixed-50pct-${tag}-s6201/answers.jsonl" \
    "${run_root}/full500-group-fixed-75pct-${tag}-s6201/answers.jsonl" \
    "${run_root}/full500-group-fixed-100pct-${tag}-s6201/answers.jsonl" \
    "${run_root}/full500-full-lclm-niah-${tag}/answers.jsonl" \
    "${run_root}/ruler-role-oracle-full500-v2-key_answer_support-25pct-${tag}-s6202/answers.jsonl"; do
    test -f "${path}"
  done
  "${python_bin}" -u -m inference.examples.analyze_joint_role_recall "${args[@]}"
done
