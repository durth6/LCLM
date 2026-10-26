#!/usr/bin/env bash
# Compare answer-span and key-answer/support oracle selections on corrected
# RULER inputs.  Both paths use the frozen Decoder and the same exact K;
# role labels are consumed only by this explicitly offline oracle ablation.
#
# Submit after role sidecars are ready:
#   sbatch --dependency=afterok:<ROLE_JOB_ID> --array=0-2 --gpus=1 -p gpu_4090 \
#     scripts/run_ruler_role_oracle_eval_array.sbatch
#
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --array=0-2
#SBATCH --time=24:00:00
#SBATCH --job-name=lclm-role-oracle
#SBATCH --output=/data/run01/scxj745/context-exp/LCLM/artifacts/logs/lclm-role-oracle-%A_%a.log

set -euo pipefail

readonly root=/data/run01/scxj745/context-exp/LCLM
readonly python_bin="${PYTHON_BIN:-/data/home/scxj745/.conda/envs/qwen35/bin/python}"
readonly checkpoint="${CHECKPOINT:-${root}/artifacts/models/0.6b-4b-LCLM-8x}"
readonly task_index="${SLURM_ARRAY_TASK_ID:-0}"
readonly seeds=(6201 6202 6203)
readonly seed="${seeds[task_index]}"
readonly max_tokens="${MAX_TARGET_TOKENS:-128}"
readonly data_tag="${RULER_DATA_TAG:-v2}"
readonly data_stem="ruler_all_tasks${data_tag:+_${data_tag}}"
readonly cache_stem="ruler-all-tasks${data_tag:+-${data_tag}}"
readonly result_prefix="${RESULT_PREFIX:-ruler-role-oracle-${data_tag:-v2}}"
readonly niah_tasks="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery"

test -n "${seed:-}"
test -f "${checkpoint}/.download_complete"

cd "${root}"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

role_input_root="$(mktemp -d "${root}/artifacts/runs/.role-oracle-input-${SLURM_JOB_ID:-local}-XXXXXX")"
trap 'rm -rf -- "${role_input_root}"' EXIT

for tag in 4k 8k 16k 32k; do
  input_dir="${root}/artifacts/data/${data_stem}_${tag}"
  cache_dir="${root}/artifacts/runs/${cache_stem}-${tag}-cache"
  test -f "${input_dir}/queries.jsonl"
  test -f "${input_dir}/oracle_latents.jsonl"
  test -f "${input_dir}/oracle_latents_roles.jsonl"
  test -f "${cache_dir}/manifest.json"

  # Role spans are intentionally defined only for the structured NIAH fact
  # tasks.  Filter by query ID for this offline diagnostic instead of asking
  # the role oracle to fabricate labels for vt/cwe/fwe/qa rows.  The temporary
  # input contains queries and sidecars only; the original source memory is
  # never copied into the answer-time path, and the existing latent cache is
  # reused directly.
  role_input_dir="${role_input_root}/${tag}"
  "${python_bin}" - "${input_dir}" "${role_input_dir}" "${niah_tasks}" <<'PY'
import json
import sys
from pathlib import Path

source_dir = Path(sys.argv[1])
target_dir = Path(sys.argv[2])
tasks = set(sys.argv[3].split(","))
target_dir.mkdir(parents=True, exist_ok=True)
queries = []
for line in (source_dir / "queries.jsonl").read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if row.get("task") in tasks:
        queries.append(row)
if not queries:
    raise SystemExit("NIAH role filter produced no queries")
ids = {str(row["id"]) for row in queries}
(target_dir / "queries.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in queries)
)
for name in ("oracle_latents.jsonl", "oracle_latents_roles.jsonl"):
    rows = []
    for line in (source_dir / name).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row["id"]) in ids:
            rows.append(row)
    if len(rows) != len(queries):
        raise SystemExit(f"{name} has {len(rows)} filtered rows for {len(queries)} queries")
    (target_dir / name).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
print(json.dumps({"queries": len(queries), "tasks": sorted(tasks)}))
PY

  for mode in answer_span key_answer_support; do
    if [[ "${mode}" = answer_span ]]; then
      oracle_jsonl="${role_input_dir}/oracle_latents.jsonl"
      oracle_role_args=()
    else
      oracle_jsonl="${role_input_dir}/oracle_latents_roles.jsonl"
      oracle_role_args=(--oracle-role key_answer,support)
    fi
    work_dir="${root}/artifacts/runs/${result_prefix}-${mode}-25pct-${tag}-s${seed}"
    eval_args=(
      --checkpoint "${checkpoint}"
      --input-dir "${role_input_dir}"
      --cache-dir "${cache_dir}"
      --work-dir "${work_dir}"
      --selection-ratio 0.25
      --selector-mode oracle
      --oracle-jsonl "${oracle_jsonl}"
      --position-policy full_layout
      --compute-answer-nll
      --max-tokens "${max_tokens}"
      --dtype bf16
      --random-seed "${seed}"
      --reuse-queries 1
      --repeat-each-query 1
      --max-resident-caches 1
      "${oracle_role_args[@]}"
    )
    # Empty array expansion above is intentional; Bash preserves all other
    # arguments while omitting --oracle-role for the answer-span control.
    "${python_bin}" -u -m inference.examples.eval_query_aware_niah "${eval_args[@]}"
  done
done
