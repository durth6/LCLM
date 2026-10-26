"""Condition RULER/NLL outcomes on joint role coverage.

This is a post-hoc diagnostic.  It reads answer JSONL and offline role
sidecars, never source text, and does not alter answer-time selection.  A
``joint`` role is counted as covered only when every numbered role group has
at least one selected latent.  The report therefore separates "some evidence
was selected" from "all key/answer/support groups were jointly selected".
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


_GROUP_RE = re.compile(r"^(key_answer|support|answer|key)_(\d+)$")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _score(row: dict[str, Any]) -> float:
    answers = [str(value).lower() for value in row.get("answers", [])]
    response = str(row.get("response", "")).lower()
    if not answers:
        return 0.0
    if row.get("scoring_function") == "ruler_string_match_part":
        return 100.0 if any(answer in response for answer in answers) else 0.0
    return 100.0 * sum(answer in response for answer in answers) / len(answers)


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.mean(values) if values else None


def _role_groups(sidecar: dict[str, Any]) -> dict[str, list[int]]:
    roles = sidecar.get("role_latent_indices", {})
    if not isinstance(roles, dict):
        return {}
    output: dict[str, list[int]] = {}
    for name, values in roles.items():
        if _GROUP_RE.fullmatch(str(name)) and isinstance(values, list):
            output[str(name)] = [int(value) for value in values]
    return output


def _joint_metrics(selected: set[int], sidecar: dict[str, Any]) -> dict[str, float | int | None]:
    groups = _role_groups(sidecar)
    key_groups = {name: values for name, values in groups.items() if name.startswith("key_answer_")}
    support_groups = {name: values for name, values in groups.items() if name.startswith("support_")}
    answer_groups = {name: values for name, values in groups.items() if name.startswith("answer_")}

    def fraction(mapping: dict[str, list[int]]) -> float | None:
        if not mapping:
            return None
        covered = sum(bool(selected.intersection(values)) for values in mapping.values())
        return covered / len(mapping)

    key_fraction = fraction(key_groups)
    support_fraction = fraction(support_groups)
    answer_fraction = fraction(answer_groups)
    active = [value for value in (key_fraction, support_fraction) if value is not None]
    joint = min(active) if active else None
    return {
        "key_answer_group_fraction": key_fraction,
        "support_group_fraction": support_fraction,
        "answer_group_fraction": answer_fraction,
        "joint_role_recall": joint,
        "key_answer_group_count": len(key_groups),
        "support_group_count": len(support_groups),
        "joint_role_complete": (joint == 1.0) if joint is not None else None,
    }


def _condition(value: float | None) -> str:
    if value is None:
        return "unavailable"
    if value >= 1.0 - 1e-12:
        return "complete"
    if value <= 1e-12:
        return "none"
    return "partial"


def analyze(answers_paths: list[tuple[str, Path]], role_path: Path) -> dict[str, Any]:
    role_rows = {str(row["id"]): row for row in _read_jsonl(role_path)}
    all_rows: list[dict[str, Any]] = []
    for label, path in answers_paths:
        for row in _read_jsonl(path):
            record_id = str(row["id"])
            sidecar = role_rows.get(record_id)
            if sidecar is None:
                raise ValueError(f"role sidecar missing {record_id} for {path}")
            selected = {int(value) for value in row.get("selected_latent_indices", [])}
            metrics = _joint_metrics(selected, sidecar)
            merged = {
                "source": label,
                "id": record_id,
                "task": row.get("task", sidecar.get("task")),
                "score": _score(row),
                "answer_nll_retention": row.get("answer_nll_retention"),
                "answer_nll_delta_vs_full": row.get("answer_nll_delta_vs_full"),
                **metrics,
            }
            all_rows.append(merged)

    conditions: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        conditions.setdefault(_condition(row["joint_role_recall"]), []).append(row)
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        by_source.setdefault(row["source"], []).append(row)
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        by_task.setdefault(str(row["task"]), []).append(row)

    def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        valid_rows = [row for row in rows if row.get("joint_role_recall") is not None]
        complete_count = sum(row.get("joint_role_complete") is True for row in valid_rows)
        return {
            "examples": len(rows),
            "role_valid_examples": len(valid_rows),
            "role_unavailable_examples": len(rows) - len(valid_rows),
            "ruler_mean": _mean(rows, "score"),
            "joint_role_recall_mean": _mean(rows, "joint_role_recall"),
            "key_answer_group_fraction_mean": _mean(rows, "key_answer_group_fraction"),
            "support_group_fraction_mean": _mean(rows, "support_group_fraction"),
            "answer_nll_retention_mean": _mean(rows, "answer_nll_retention"),
            "selected_full_gap_mean": _mean(rows, "answer_nll_delta_vs_full"),
            "complete_joint_count": complete_count,
            "complete_joint_fraction": (
                complete_count / len(valid_rows) if valid_rows else None
            ),
        }

    return {
        "schema_version": 2,
        "joint_definition": "min(fraction of numbered key_answer groups covered, fraction of numbered support groups covered); unavailable when no numbered groups exist",
        "fraction_denominator": "role-valid examples only; unavailable rows are excluded",
        "role_sidecar": str(role_path),
        "sources": {name: summary(rows) for name, rows in sorted(by_source.items())},
        "conditions": {name: summary(rows) for name, rows in sorted(conditions.items())},
        "by_task": {name: summary(rows) for name, rows in sorted(by_task.items())},
        "rows": all_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", action="append", required=True, help="LABEL=answers.jsonl")
    parser.add_argument("--role-jsonl", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    sources: list[tuple[str, Path]] = []
    for value in args.answers:
        if "=" not in value:
            raise ValueError("--answers values must use LABEL=PATH")
        label, raw_path = value.split("=", 1)
        if not label or not raw_path:
            raise ValueError("--answers values must use non-empty LABEL=PATH")
        sources.append((label, Path(raw_path)))
    report = analyze(sources, Path(args.role_jsonl))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"out": str(output), "rows": len(report["rows"])}))


if __name__ == "__main__":
    main()
