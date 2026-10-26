#!/usr/bin/env bash
# Build offline NIAH key/answer/support role labels for the corrected v2 data.
# Role labels are diagnostic/oracle sidecars only; they are never read by the
# latent-only online selector unless an explicitly requested oracle ablation
# is being evaluated.
#
# Submit with one independent 4090 per context length:
#   sbatch --array=0-3 --gpus=1 -p gpu_4090 scripts/run_ruler_roles_array.sbatch
#
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --array=0-3
# Full 500-example/task role annotation scales with the 6,500 rows per
# length; keep enough headroom for the 32K GPU oracle pass.
#SBATCH --time=08:00:00
#SBATCH --job-name=lclm-ruler-roles
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-ruler-roles-%A_%a.log

set -euo pipefail

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly python_bin="${PYTHON_BIN:-/data/home/scxj745/.conda/envs/qwen35/bin/python}"
readonly checkpoint="${CHECKPOINT:-${root}/artifacts/models/0.6b-4b-LCLM-8x}"
readonly task_index="${SLURM_ARRAY_TASK_ID:-0}"
readonly data_tag="${RULER_DATA_TAG:-v2}"

case "${task_index}" in
  0) tag=4k ;;
  1) tag=8k ;;
  2) tag=16k ;;
  3) tag=32k ;;
  *) echo "unsupported array index ${task_index}" >&2; exit 2 ;;
esac

readonly data_stem="ruler_all_tasks_${data_tag}"
readonly cache_stem="ruler-all-tasks-${data_tag}"
readonly input_dir="${root}/artifacts/data/${data_stem}_${tag}"
readonly cache_dir="${root}/artifacts/runs/${cache_stem}-${tag}-cache"
readonly role_output="${ROLE_OUTPUT:-${input_dir}/role_spans.jsonl}"
readonly oracle_output="${ORACLE_OUTPUT:-${input_dir}/oracle_latents_roles.jsonl}"
readonly support_radius="${SUPPORT_RADIUS:-96}"
readonly role_tmp="${role_output}.tmp-${SLURM_JOB_ID:-local}"
readonly oracle_tmp="${oracle_output}.tmp-${SLURM_JOB_ID:-local}"

test -f "${checkpoint}/.download_complete"
test -f "${input_dir}/memory.jsonl"
test -f "${input_dir}/queries.jsonl"
test -f "${cache_dir}/manifest.json"
mkdir -p "${input_dir}"

cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

query_count="$(wc -l < "${input_dir}/queries.jsonl")"
if [[ -f "${role_output}" ]] && [[ "$(wc -l < "${role_output}")" -eq "${query_count}" ]]; then
  echo "reusing complete role sidecar ${role_output}"
else
  rm -f "${role_tmp}"
  "${python_bin}" -u -m inference.examples.annotate_ruler_roles \
    --memory-jsonl "${input_dir}/memory.jsonl" \
    --queries-jsonl "${input_dir}/queries.jsonl" \
    --out "${role_tmp}" \
    --support-radius "${support_radius}"
  test "$(wc -l < "${role_tmp}")" -eq "${query_count}"
  mv -f "${role_tmp}" "${role_output}"
fi

if [[ -f "${oracle_output}" ]] && [[ "$(wc -l < "${oracle_output}")" -eq "${query_count}" ]]; then
  echo "reusing complete role-aware oracle sidecar ${oracle_output}"
else
  rm -f "${oracle_tmp}"
  "${python_bin}" -u -m inference.examples.annotate_oracle_latents \
    --checkpoint "${checkpoint}" \
    --memory-jsonl "${input_dir}/memory.jsonl" \
    --queries-jsonl "${input_dir}/queries.jsonl" \
    --cache-dir "${cache_dir}" \
    --out "${oracle_tmp}" \
    --role-spans-jsonl "${role_output}"
  test "$(wc -l < "${oracle_tmp}")" -eq "${query_count}"
  mv -f "${oracle_tmp}" "${oracle_output}"
fi

"${python_bin}" - "${oracle_output}" <<'PY'
import json
import sys

path = sys.argv[1]
rows = [json.loads(line) for line in open(path) if line.strip()]
available = [row for row in rows if row.get("role_labels_available")]
if not available:
    raise SystemExit("role-aware oracle contains no available role labels")
print(json.dumps({
    "records": len(rows),
    "role_labels_available": len(available),
    "path": path,
}, indent=2))
PY
