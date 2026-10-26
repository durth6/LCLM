"""Aggregate audited Full500 fixed-budget summaries across optimization seeds."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = (
    "ruler_accuracy",
    "answer_nll_retention_mean",
    "answer_nll_delta_vs_full_mean",
    "answer_latent_recall_mean",
    "all_answer_latent_recall_mean",
    "key_answer_joint_recall_mean",
    "support_latent_recall_mean",
    "selection_ratio_actual_mean",
    "k_decoder_mean",
    "decoder_kv_cache_bytes_estimate_mean",
    "selector_latency_ms_mean",
    "adapter_latency_ms_mean",
    "decoder_prefill_latency_ms_mean",
    "cache_resident_ttft_ms_mean",
)


def _mean_sd(values: list[float]) -> dict[str, float | int | None]:
    return {
        "mean": statistics.mean(values) if values else None,
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
        "count": len(values),
    }


def _parse_budget(value: str) -> tuple[float, str]:
    if "=" not in value:
        raise ValueError("--budget must use RATIO=PATH_TEMPLATE")
    raw_ratio, template = value.split("=", 1)
    ratio = float(raw_ratio)
    if not 0.0 < ratio <= 1.0 or not template:
        raise ValueError("budget ratio must be in (0, 1] and template must be nonempty")
    if "{tag}" not in template or "{seed}" not in template:
        raise ValueError("budget path template must contain {tag} and {seed}")
    return ratio, template


def aggregate(
    run_root: Path,
    budgets: list[tuple[float, str]],
    tags: list[str],
    seeds: list[int],
    expected_queries: int,
) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for ratio, template in sorted(budgets):
        for tag in tags:
            summaries = []
            sources = []
            for seed in seeds:
                source = run_root / template.format(tag=tag, seed=seed) / "summary.json"
                if not source.is_file():
                    raise FileNotFoundError(source)
                summary = json.loads(source.read_text())
                if int(summary.get("num_queries", -1)) != expected_queries:
                    raise ValueError(f"{source}: expected {expected_queries} queries")
                if summary.get("position_policy") != "full_layout":
                    raise ValueError(f"{source}: sparse position policy is not full_layout")
                actual = summary.get("selection_ratio_actual_mean")
                if actual is None or abs(float(actual) - ratio) > 0.01:
                    raise ValueError(f"{source}: actual selection ratio {actual} mismatches {ratio}")
                summaries.append(summary)
                sources.append(str(source))

            metric_summary = {}
            for metric in METRICS:
                values = [float(row[metric]) for row in summaries if row.get(metric) is not None]
                metric_summary[metric] = _mean_sd(values)
            tasks = sorted({task for row in summaries for task in row.get("per_task_accuracy", {})})
            per_task = {
                task: _mean_sd([
                    float(row["per_task_accuracy"][task])
                    for row in summaries if task in row.get("per_task_accuracy", {})
                ])
                for task in tasks
            }
            points.append({
                "budget_ratio": ratio,
                "length": tag,
                "seeds": seeds,
                "sources": sources,
                "metrics": metric_summary,
                "per_task_accuracy": per_task,
            })
    return {
        "schema_version": 1,
        "scope": "Full500 fixed-budget cost-quality Pareto; three optimization seeds",
        "expected_queries_per_seed_length": expected_queries,
        "position_policy": "full_layout",
        "points": points,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="artifacts/runs")
    parser.add_argument("--budget", action="append", required=True)
    parser.add_argument("--tags", default="4k,8k,16k,32k")
    parser.add_argument("--seeds", default="6201,6202,6203")
    parser.add_argument("--expected-queries", type=int, default=6500)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = aggregate(
        Path(args.run_root),
        [_parse_budget(value) for value in args.budget],
        [value for value in args.tags.split(",") if value],
        [int(value) for value in args.seeds.split(",") if value],
        args.expected_queries,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"out": str(output), "points": len(report["points"])}))


if __name__ == "__main__":
    main()
