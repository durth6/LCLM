"""Aggregate Full500 calibrated dynamic-K results against a matched fixed-K run."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


METRICS = (
    "ruler_accuracy",
    "selection_ratio_actual_mean",
    "answer_nll_retention_mean",
    "answer_nll_delta_vs_full_mean",
    "key_answer_joint_recall_mean",
    "support_latent_recall_mean",
    "decoder_kv_cache_bytes_estimate_mean",
    "decoder_prefill_latency_ms_mean",
    "cache_resident_ttft_ms_mean",
    "selector_latency_ms_mean",
)

DISCRETE_RATIOS = (0.25, 0.5, 0.75, 1.0)


def _mean_sd(values: list[float]) -> dict[str, float | int | None]:
    return {
        "mean": statistics.mean(values) if values else None,
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
        "count": len(values),
    }


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _audit_answers(path: Path, expected_queries: int) -> dict[str, Any]:
    count = 0
    ids: set[str] = set()
    ratio_values: list[float] = []
    budget_counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    for row in _read_jsonl(path):
        count += 1
        row_id = str(row["id"])
        if row_id in ids:
            raise ValueError(f"{path}: duplicate query ID {row_id}")
        ids.add(row_id)
        indices = [int(value) for value in row.get("selected_latent_indices", [])]
        if indices != sorted(indices):
            raise ValueError(f"{path}: selected indices are not source ordered for {row_id}")
        if row.get("position_policy") != "full_layout":
            raise ValueError(f"{path}: position policy is not full_layout")
        if row.get("answer_nll_selected") is None:
            raise ValueError(f"{path}: answer NLL missing for {row_id}")
        ratio = float(row["selection_ratio"])
        ratio_values.append(ratio)
        # K/M is rounded by ceil(K/M), so a 25% selection may be a few
        # ulps above .25. Report the declared discrete operating point.
        declared_ratio = min(DISCRETE_RATIOS, key=lambda value: abs(value - ratio))
        if abs(declared_ratio - ratio) > 0.02:
            raise ValueError(f"{path}: row ratio {ratio} is outside the discrete budget grid")
        budget_counts[f"{declared_ratio:.12g}"] += 1
        policy_counts[str(row.get("selection_policy"))] += 1
    if count != expected_queries:
        raise ValueError(f"{path}: expected {expected_queries} rows, found {count}")
    return {
        "queries": count,
        "selection_ratio_mean_from_rows": statistics.mean(ratio_values) if ratio_values else None,
        "budget_distribution": dict(sorted(budget_counts.items(), key=lambda item: float(item[0]))),
        "selection_policy_distribution": dict(policy_counts),
    }


def _paired_audit(
    dynamic_path: Path,
    fixed_path: Path,
    expected_queries: int,
    *,
    require_exact_selection: bool,
) -> dict[str, Any]:
    """Verify paired IDs/contracts without retaining large selected sets in RAM."""
    count = 0
    exact_selection_count = 0
    exact_response_count = 0
    overlap_values: list[float] = []
    with dynamic_path.open() as dynamic_handle, fixed_path.open() as fixed_handle:
        for dynamic_line, fixed_line in itertools.zip_longest(dynamic_handle, fixed_handle):
            if dynamic_line is None or fixed_line is None:
                raise ValueError("paired answer JSONL row counts differ")
            if not dynamic_line.strip() or not fixed_line.strip():
                raise ValueError("paired answer JSONL must not contain blank rows")
            dynamic = json.loads(dynamic_line)
            fixed = json.loads(fixed_line)
            count += 1
            if dynamic.get("id") != fixed.get("id"):
                raise ValueError(f"paired query ID mismatch at row {count}")
            if dynamic.get("document_id") != fixed.get("document_id"):
                raise ValueError(f"paired document ID mismatch at row {count}")
            if dynamic.get("cache_fingerprint") != fixed.get("cache_fingerprint"):
                raise ValueError(f"paired cache fingerprint mismatch at row {count}")
            dynamic_indices = set(int(value) for value in dynamic["selected_latent_indices"])
            fixed_indices = set(int(value) for value in fixed["selected_latent_indices"])
            exact = dynamic_indices == fixed_indices
            exact_selection_count += int(exact)
            exact_response_count += int(dynamic.get("response") == fixed.get("response"))
            union = dynamic_indices | fixed_indices
            overlap_values.append(len(dynamic_indices & fixed_indices) / len(union) if union else 1.0)
            if require_exact_selection and not exact:
                raise ValueError(f"fallback selection differs from fixed 50% at row {count}")
    if count != expected_queries:
        raise ValueError(f"paired answer JSONL expected {expected_queries} rows, found {count}")
    return {
        "queries": count,
        "exact_selected_fraction": exact_selection_count / count if count else None,
        "exact_response_fraction": exact_response_count / count if count else None,
        "selected_set_jaccard_mean": statistics.mean(overlap_values) if overlap_values else None,
        "fallback_exact_selection_required": require_exact_selection,
    }


def _load_summary(path: Path, expected_queries: int) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = path / "summary.json"
    answers_path = path / "answers.jsonl"
    if not summary_path.is_file() or not answers_path.is_file():
        raise FileNotFoundError(path)
    summary = json.loads(summary_path.read_text())
    if int(summary.get("num_queries", -1)) != expected_queries:
        raise ValueError(f"{summary_path}: wrong query count")
    if summary.get("position_policy") != "full_layout":
        raise ValueError(f"{summary_path}: wrong position policy")
    return summary, _audit_answers(answers_path, expected_queries)


def _aggregate_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in METRICS:
        output[metric] = _mean_sd([
            float(summary[metric])
            for summary in summaries
            if summary.get(metric) is not None
        ])
    tasks = sorted({
        task
        for summary in summaries
        for task in summary.get("per_task_accuracy", {})
    })
    output["per_task_accuracy"] = {
        task: _mean_sd([
            float(summary["per_task_accuracy"][task])
            for summary in summaries
            if task in summary.get("per_task_accuracy", {})
        ])
        for task in tasks
    }
    return output


def _path(template: str, root: Path, tag: str, seed: int) -> Path:
    rendered = template.format(tag=tag, seed=seed)
    path = Path(rendered)
    return path if path.is_absolute() else root / path


def aggregate(
    dynamic_template: str,
    fixed_template: str,
    root: Path,
    tags: list[str],
    seeds: list[int],
    expected_queries: int,
) -> dict[str, Any]:
    lengths: list[dict[str, Any]] = []
    for tag in tags:
        dynamic_summaries: list[dict[str, Any]] = []
        fixed_summaries: list[dict[str, Any]] = []
        dynamic_audits: list[dict[str, Any]] = []
        fixed_audits: list[dict[str, Any]] = []
        sources: list[dict[str, Any]] = []
        for seed in seeds:
            dynamic_path = _path(dynamic_template, root, tag, seed)
            fixed_path = _path(fixed_template, root, tag, seed)
            dynamic_summary, dynamic_audit = _load_summary(dynamic_path, expected_queries)
            fixed_summary, fixed_audit = _load_summary(fixed_path, expected_queries)
            if not bool(dynamic_summary.get("dynamic_k")):
                raise ValueError(f"{dynamic_path}: dynamic_k is not enabled")
            if abs(float(fixed_summary.get("selection_ratio_actual_mean", 0.0)) - 0.5) > 0.01:
                raise ValueError(f"{fixed_path}: fixed comparison is not 50%")
            paired_audit = _paired_audit(
                dynamic_path / "answers.jsonl",
                fixed_path / "answers.jsonl",
                expected_queries,
                require_exact_selection=(
                    dynamic_summary.get("dynamic_calibration_fallback_ratio") == 0.5
                ),
            )
            dynamic_summaries.append(dynamic_summary)
            fixed_summaries.append(fixed_summary)
            dynamic_audits.append(dynamic_audit)
            fixed_audits.append(fixed_audit)
            sources.append({
                "seed": seed,
                "dynamic": str(dynamic_path),
                "fixed": str(fixed_path),
                "paired_audit": paired_audit,
            })
        dynamic_aggregate = _aggregate_summaries(dynamic_summaries)
        fixed_aggregate = _aggregate_summaries(fixed_summaries)
        deltas: dict[str, Any] = {}
        for metric in METRICS:
            dynamic_value = dynamic_aggregate[metric]["mean"]
            fixed_value = fixed_aggregate[metric]["mean"]
            deltas[metric] = (
                dynamic_value - fixed_value
                if dynamic_value is not None and fixed_value is not None
                else None
            )
        lengths.append({
            "length": tag,
            "seeds": seeds,
            "sources": sources,
            "dynamic": dynamic_aggregate,
            "fixed_50": fixed_aggregate,
            "delta_dynamic_minus_fixed_50": deltas,
            "dynamic_row_audits": dynamic_audits,
            "fixed_row_audits": fixed_audits,
            "calibration": [
                {
                    "seed": seed,
                    "policy": dynamic_summary.get("dynamic_calibration_policy"),
                    "temperature": dynamic_summary.get("dynamic_calibration_temperature"),
                    "qualified_against_fixed_50": dynamic_summary.get(
                        "dynamic_calibration_qualified_against_fixed_50"
                    ),
                    "fallback_ratio": dynamic_summary.get("dynamic_calibration_fallback_ratio"),
                }
                for seed, dynamic_summary in zip(seeds, dynamic_summaries)
            ],
        })
    return {
        "schema_version": 1,
        "scope": "Full500 calibrated dynamic-K versus matched role-aware fixed 50%; three seeds",
        "position_policy": "full_layout",
        "expected_queries_per_seed_length": expected_queries,
        "dynamic_policy": "role_score_mass_v1 with validation-only calibration and guarded fixed-50 fallback",
        "lengths": lengths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/runs")
    parser.add_argument(
        "--dynamic-template",
        default="full500-role-aware-calibrated-dynamic-k-{tag}-s{seed}",
    )
    parser.add_argument(
        "--fixed-template",
        default="full500-role-aware-fixed-50pct-{tag}-s{seed}",
    )
    parser.add_argument("--tags", default="4k,8k,16k,32k")
    parser.add_argument("--seeds", default="6201,6202,6203")
    parser.add_argument("--expected-queries", type=int, default=6500)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = aggregate(
        args.dynamic_template,
        args.fixed_template,
        Path(args.root),
        [value for value in args.tags.split(",") if value],
        [int(value) for value in args.seeds.split(",") if value],
        args.expected_queries,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"out": str(output), "lengths": len(report["lengths"])}))


if __name__ == "__main__":
    main()
