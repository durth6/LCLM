#!/usr/bin/env bash
# Option 1: strict full-context/full-latent baseline on the 104-row v2 16K set.
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --job-name=lclm-v2-full16k
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-v2-full16k-%j.log

set -euo pipefail

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly python_bin=/data/home/scxj745/.conda/envs/qwen35/bin/python
readonly checkpoint=${root}/artifacts/models/0.6b-4b-LCLM-8x
readonly input_dir=${root}/artifacts/data/ruler_all_tasks_v2_16k
readonly cache_dir=${root}/artifacts/runs/ruler-all-tasks-v2-16k-cache
readonly work_dir=${root}/artifacts/runs/full-100pct-v2-16k

test -f "${checkpoint}/.download_complete"
test -f "${input_dir}/queries.jsonl"
test -f "${cache_dir}/manifest.json"
test ! -e "${work_dir}/answers.jsonl"

cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

exec "${python_bin}" -u -m inference.examples.eval_query_aware_niah \
  --checkpoint "${checkpoint}" \
  --input-dir "${input_dir}" \
  --cache-dir "${cache_dir}" \
  --work-dir "${work_dir}" \
  --selection-ratio 1.0 \
  --selector-mode full \
  --position-policy full_layout \
  --compute-answer-nll \
  --max-examples 104 \
  --max-tokens 128 \
  --dtype bf16 \
  --random-seed 6201 \
  --reuse-queries 1 \
  --repeat-each-query 1 \
  --max-resident-caches 1
