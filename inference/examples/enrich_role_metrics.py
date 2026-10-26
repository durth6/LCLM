"""Join offline role labels onto an existing query-aware answer run.

This utility is intentionally post-hoc: it reads only ``answers.jsonl`` and
the offline role-aware oracle sidecar.  It never opens source memory or
changes the selected latent set, so historical evaluations can gain the same
role-recall diagnostics as newly generated runs without another GPU pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

from inference.examples.eval_query_aware_niah import _attach_role_metrics, _read_jsonl


ROLE_FIELDS = (
    "answer_latent_recall",
    "all_answer_latent_recall",
    "key_answer_joint_recall",
    "support_latent_recall",
)


def _role_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for field in ROLE_FIELDS:
        values = [
            float(record[field])
            for record in records
            if record.get(field) is not None
        ]
        summary[f"{field}_mean"] = statistics.mean(values) if values else None
        summary[f"{field}_valid_count"] = len(values)

    per_task: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        per_task.setdefault(str(record.get("task", "<unknown>")), []).append(record)
    # Keep task-level values nested so unavailable role labels remain visibly
    # distinct from a measured zero recall.
    summary["per_task_role_metrics"] = {}
    for task, task_records in per_task.items():
        task_summary: dict[str, Any] = {}
        for field in ROLE_FIELDS:
            values = [
                float(record[field])
                for record in task_records
                if record.get(field) is not None
            ]
            task_summary[f"{field}_mean"] = statistics.mean(values) if values else None
            task_summary[f"{field}_valid_count"] = len(values)
        summary["per_task_role_metrics"][task] = task_summary
    return summary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich an existing answer JSONL with offline role recall metrics"
    )
    parser.add_argument("--answers-jsonl", required=True)
    parser.add_argument("--roles-jsonl", required=True)
    parser.add_argument("--out-answers", required=True)
    parser.add_argument(
        "--summary-in",
        help="optional existing summary.json to update with the joined metrics",
    )
    parser.add_argument(
        "--summary-out",
        help="output summary path; defaults to summary-with-roles.json beside answers",
    )
    args = parser.parse_args()

    answers_path = Path(args.answers_jsonl)
    out_answers = Path(args.out_answers)
    records = _read_jsonl(answers_path)
    if not records:
        raise ValueError(f"answer JSONL is empty: {answers_path}")
    enriched = _attach_role_metrics(records, args.roles_jsonl)
    _write_jsonl(out_answers, enriched)

    role_summary = _role_summary(enriched)
    output = {
        "role_metrics_schema_version": 1,
        "role_sidecar": str(Path(args.roles_jsonl)),
        "answer_records": len(enriched),
        **role_summary,
    }
    if args.summary_in:
        summary_input = Path(args.summary_in)
        base = json.loads(summary_input.read_text())
        if not isinstance(base, dict):
            raise ValueError(f"summary must be a JSON object: {summary_input}")
        base.update(role_summary)
        base.update({
            "role_metrics_schema_version": 1,
            "role_sidecar": str(Path(args.roles_jsonl)),
            "role_metrics_joined_from": str(answers_path),
        })
        output = base
    summary_out = args.summary_out
    if summary_out is None and args.summary_in:
        summary_out = str(Path(args.summary_in).with_name("summary-with-roles.json"))
    if summary_out:
        summary_path = Path(summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps({
        "answer_records": len(enriched),
        **{
            key: role_summary[key]
            for key in (
                "answer_latent_recall_mean",
                "all_answer_latent_recall_mean",
                "key_answer_joint_recall_mean",
                "support_latent_recall_mean",
            )
        },
    }, indent=2))


if __name__ == "__main__":
    main()
