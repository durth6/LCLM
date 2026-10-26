"""Audit a learned selector's selected latent sets against an oracle sidecar."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _read(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _histogram(indices: set[int], memory_size: int, buckets: int = 8) -> list[float]:
    counts = [0] * buckets
    if memory_size <= 0:
        return [0.0] * buckets
    for index in indices:
        bucket = min(buckets - 1, index * buckets // memory_size)
        counts[bucket] += 1
    total = sum(counts)
    return [count / total if total else 0.0 for count in counts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers-jsonl", required=True)
    parser.add_argument("--oracle-jsonl", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    answers = {str(row["id"]): row for row in _read(args.answers_jsonl)}
    oracle = {str(row["id"]): row for row in _read(args.oracle_jsonl)}
    if set(answers) != set(oracle):
        raise ValueError("answer and oracle IDs do not match")

    rows = []
    per_task: dict[str, list[dict[str, float]]] = defaultdict(list)
    selected_hist = [0.0] * 8
    oracle_hist = [0.0] * 8
    for record_id, answer in answers.items():
        selected = {int(index) for index in answer.get("selected_latent_indices", [])}
        oracle_row = oracle[record_id]
        oracle_indices = {
            int(index)
            for index in oracle_row.get("oracle_latent_indices", [])
        }
        memory_size = int(answer["cached_latent_count"])
        if list(answer.get("selected_latent_indices", [])) != sorted(selected):
            raise ValueError(f"selected indices are not source ordered for {record_id}")
        overlap = selected & oracle_indices
        union = selected | oracle_indices
        metrics = {
            "precision": len(overlap) / len(selected) if selected else 0.0,
            "recall": len(overlap) / len(oracle_indices) if oracle_indices else 0.0,
            "jaccard": len(overlap) / len(union) if union else 1.0,
            "selected_fraction": len(selected) / memory_size,
        }
        rows.append({"id": record_id, "task": answer.get("task"), **metrics})
        per_task[str(answer.get("task", "unknown"))].append(metrics)
        selected_bucket = _histogram(selected, memory_size)
        oracle_bucket = _histogram(oracle_indices, memory_size)
        selected_hist = [left + right for left, right in zip(selected_hist, selected_bucket)]
        oracle_hist = [left + right for left, right in zip(oracle_hist, oracle_bucket)]

    def aggregate(items: list[dict[str, float]]) -> dict[str, float]:
        numeric_keys = [
            key for key, value in items[0].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        return {key: mean(float(item[key]) for item in items) for key in numeric_keys}

    result = {
        "schema_version": 1,
        "answers_jsonl": str(Path(args.answers_jsonl)),
        "oracle_jsonl": str(Path(args.oracle_jsonl)),
        "num_queries": len(rows),
        "mean": aggregate(rows),
        "per_task": {task: aggregate(items) for task, items in sorted(per_task.items())},
        "selected_position_histogram_mean": [value / len(rows) for value in selected_hist],
        "oracle_position_histogram_mean": [value / len(rows) for value in oracle_hist],
        "source_order_validated": True,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
