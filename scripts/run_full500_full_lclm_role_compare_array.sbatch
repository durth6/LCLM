#!/usr/bin/env bash
# Full-latent LCLM control on the same Full500 NIAH rows used by the role oracle.
# The source context is never passed to answer-time; only the prebuilt latent
# cache and filtered selector queries are used.
#
# Submit after the fixed-budget Full500 stage:
#   sbatch --dependency=afterok:<100P_JOB> --array=0-2 --gpus=1 -p gpu_4090 \
#     scripts/run_full500_full_lclm_role_compare_array.sbatch
#
# Array indices: 0=8K, 1=16K, 2=32K.
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --array=0-2
#SBATCH --time=24:00:00
#SBATCH --job-name=lclm-full-role-compare
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-full-role-compare-%A_%a.log

set -euo pipefail

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly python_bin="${PYTHON_BIN:-/data/home/scxj745/.conda/envs/qwen35/bin/python}"
readonly checkpoint="${CHECKPOINT:-${root}/artifacts/models/0.6b-4b-LCLM-8x}"
readonly task_index="${SLURM_ARRAY_TASK_ID:-0}"
readonly tags=(8k 16k 32k)
readonly tag="${tags[task_index]}"
readonly data_tag="${RULER_DATA_TAG:-full500-v2}"
readonly data_stem="ruler_all_tasks_${data_tag}"
readonly cache_stem="ruler-all-tasks-${data_tag}"
readonly input_dir="${root}/artifacts/data/${data_stem}_${tag}"
readonly cache_dir="${root}/artifacts/runs/${cache_stem}-${tag}-cache"
readonly result_dir="${root}/artifacts/runs/${RESULT_PREFIX:-full500-full-lclm-niah}-${tag}"
readonly niah_tasks="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery"

if (( task_index < 0 || task_index >= ${#tags[@]} )); then
  echo "unsupported SLURM_ARRAY_TASK_ID=${task_index}" >&2
  exit 2
fi
test -f "${checkpoint}/.download_complete"
test -f "${input_dir}/queries.jsonl"
test -f "${input_dir}/oracle_latents_roles.jsonl"
test -f "${cache_dir}/manifest.json"
test ! -e "${result_dir}/answers.jsonl"
test ! -e "${result_dir}/summary.json"

cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

readonly filtered_root="$(mktemp -d "${root}/artifacts/runs/.full-role-input-${SLURM_JOB_ID:-local}-XXXXXX")"
trap 'rm -rf -- "${filtered_root}"' EXIT
readonly filtered_dir="${filtered_root}/${tag}"

"${python_bin}" - "${input_dir}" "${filtered_dir}" "${niah_tasks}" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
tasks = set(sys.argv[3].split(","))
target.mkdir(parents=True, exist_ok=True)
queries = [
    json.loads(line) for line in (source / "queries.jsonl").read_text().splitlines()
    if line.strip() and json.loads(line).get("task") in tasks
]
if not queries:
    raise SystemExit("NIAH filter produced no queries")
ids = {str(row["id"]) for row in queries}
(target / "queries.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in queries)
)
for name in ("oracle_latents.jsonl", "oracle_latents_roles.jsonl"):
    rows = [
        json.loads(line) for line in (source / name).read_text().splitlines()
        if line.strip() and str(json.loads(line)["id"]) in ids
    ]
    if len(rows) != len(queries):
        raise SystemExit(f"{name}: {len(rows)} rows for {len(queries)} queries")
    (target / name).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
print(json.dumps({"queries": len(queries), "tag": target.name}))
PY

exec "${python_bin}" -u -m inference.examples.eval_query_aware_niah \
  --checkpoint "${checkpoint}" \
  --input-dir "${filtered_dir}" \
  --cache-dir "${cache_dir}" \
  --work-dir "${result_dir}" \
  --selection-ratio 1.0 \
  --selector-mode full \
  --oracle-jsonl "${filtered_dir}/oracle_latents_roles.jsonl" \
  --role-jsonl "${filtered_dir}/oracle_latents_roles.jsonl" \
  --position-policy full_layout \
  --compute-answer-nll \
  --max-tokens 128 \
  --dtype bf16 \
  --random-seed 6201 \
  --reuse-queries 1 \
  --repeat-each-query 1 \
  --max-resident-caches 1
