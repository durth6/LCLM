#!/usr/bin/env bash
# Validate the direct answer-NLL utility oracle and its inference-side contract.
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --job-name=lclm-util-oracle
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-util-oracle-%j.log

set -euo pipefail

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly python_bin=/data/home/scxj745/.conda/envs/qwen35/bin/python
readonly checkpoint=${root}/artifacts/models/0.6b-4b-LCLM-8x
readonly input_dir=${root}/artifacts/data/ruler_niah_4k_smoke
readonly cache_dir=${root}/artifacts/runs/smoke-k256/cache
readonly learned_answers=${root}/artifacts/runs/p1-contribution-full-25pct-ctx4k-s6201/answers.jsonl
readonly oracle_out=${root}/artifacts/data/ruler_niah_4k_smoke/utility-oracle-group8-full-layout-1example.jsonl

test -f "${cache_dir}/manifest.json"
test -f "${input_dir}/oracle_latents.jsonl"
test -f "${learned_answers}"
cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${python_bin}" -u -m inference.examples.annotate_utility_oracle \
  --checkpoint "${checkpoint}" \
  --cache-dir "${cache_dir}" \
  --queries-jsonl "${input_dir}/queries.jsonl" \
  --out "${oracle_out}" \
  --selection-ratio 0.25 \
  --num-groups 8 \
  --span-oracle-jsonl "${input_dir}/oracle_latents.jsonl" \
  --candidate-jsonl "learned_p1=${learned_answers}" \
  --position-policy full_layout \
  --max-examples 1 \
  --max-target-tokens 32 \
  --dtype bf16

exec "${python_bin}" -u -m inference.examples.eval_query_aware_niah \
  --checkpoint "${checkpoint}" \
  --input-dir "${input_dir}" \
  --cache-dir "${cache_dir}" \
  --work-dir "${root}/artifacts/runs/utility-oracle-group8-full-layout-4k-1example" \
  --selection-ratio 0.25 \
  --selector-mode utility_oracle \
  --utility-oracle-jsonl "${oracle_out}" \
  --oracle-jsonl "${input_dir}/oracle_latents.jsonl" \
  --position-policy full_layout \
  --compute-answer-nll \
  --max-examples 1 \
  --max-tokens 32 \
  --dtype bf16
