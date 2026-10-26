#!/usr/bin/env bash
# Submit after the local checkpoint and prepared NIAH files exist:
#   sbatch scripts/run_query_aware_niah_8x.sbatch
# Override defaults, e.g. K_DECODER=64 LIMIT_TAG=smoke-k64 sbatch ...
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=lclm-qa-niah-8x
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-qa-niah-%j.log

set -euo pipefail

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly checkpoint="${root}/artifacts/models/0.6b-4b-LCLM-8x"
readonly input_dir="${NIAH_INPUT_DIR:-${root}/artifacts/data/ruler_niah_4k_smoke}"
readonly k_decoder="${K_DECODER:-256}"
readonly tag="${LIMIT_TAG:-smoke-k${k_decoder}}"
readonly work_dir="${root}/artifacts/runs/${tag}"

test -f "${checkpoint}/.download_complete"
test -f "${input_dir}/memory.jsonl"
test -f "${input_dir}/queries.jsonl"

cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
exec /data/home/scxj745/.conda/envs/qwen35/bin/python -u \
  -m inference.examples.eval_query_aware_niah \
  --checkpoint "${checkpoint}" \
  --input-dir "${input_dir}" \
  --work-dir "${work_dir}" \
  --k-decoder "${k_decoder}" \
  --max-tokens 128 \
  --dtype bf16 \
  --reuse-queries 1
