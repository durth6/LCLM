"""Run and score a cache-reuse NIAH experiment for query-aware LCLM.

Input must be prepared by ``prepare_ruler_niah_query_aware``.  The cache stage
is deliberately a separate process from query answering so the reported
``encoder_latency_ms`` can be amortized over the requested number of queries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any


def _ruler_score(
    answers: list[str], response: str, scoring_function: str = "ruler_string_match_all",
) -> float:
    if not answers:
        return 0.0
    response = response.lower()
    matches = [answer.lower() in response for answer in answers]
    if scoring_function == "ruler_string_match_part":
        return float(any(matches)) * 100.0
    if scoring_function != "ruler_string_match_all":
        raise ValueError(f"unsupported RULER scoring function: {scoring_function}")
    return sum(matches) / len(matches) * 100.0


def _run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _mean(records: list[dict[str, Any]], field: str) -> float:
    values = [float(record[field]) for record in records if record.get(field) is not None]
    return statistics.mean(values) if values else 0.0


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL rows while keeping sidecar handling deterministic."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _load_role_sidecar(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load offline role labels keyed by query ID.

    Role sidecars are deliberately kept separate from answer records.  The
    loader accepts both the current role-aware oracle rows and older rows with
    only ``answer_latent_indices``/``oracle_latent_indices`` so that historical
    runs can be enriched without rerunning inference.
    """
    rows: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        if "id" not in row:
            raise ValueError(f"role sidecar row is missing id: {row!r}")
        record_id = str(row["id"])
        if record_id in rows:
            raise ValueError(f"duplicate role sidecar row for {record_id}")
        schema = row.get("oracle_schema_version")
        if schema not in (None, 1):
            raise ValueError(f"unsupported role sidecar schema for {record_id}: {schema}")
        role_mapping = row.get("role_latent_indices", {})
        if role_mapping is not None and not isinstance(role_mapping, dict):
            raise ValueError(f"role_latent_indices for {record_id} must be an object")
        normalized: dict[str, list[int]] = {}
        for role, indices in (role_mapping or {}).items():
            if not isinstance(indices, list):
                raise ValueError(f"role {role!r} for {record_id} must be a list")
            normalized[str(role)] = [int(index) for index in indices]
        # Keep the historical fallback semantics: answer/all-answer recall is
        # available even when a row has no richer key/support annotations.
        if "answer" not in normalized:
            normalized["answer"] = [
                int(index) for index in row.get(
                    "answer_latent_indices", row.get("oracle_latent_indices", [])
                )
            ]
        if "all_answer" not in normalized:
            normalized["all_answer"] = [
                int(index) for index in row.get(
                    "all_answer_latent_indices", normalized["answer"]
                )
            ]
        rows[record_id] = {
            "document_id": row.get("document_id", row.get("cache_id", record_id)),
            "cache_fingerprint": row.get("cache_fingerprint"),
            "cache_schema_version": row.get("cache_schema_version"),
            "role_labels_available": row.get("role_labels_available"),
            "roles": normalized,
        }
    return rows


def _attach_role_metrics(
    records: list[dict[str, Any]], role_jsonl: str | Path | None,
) -> list[dict[str, Any]]:
    """Join offline role labels onto generated answer records.

    This is also used by the post-hoc enrichment command for runs produced
    before the evaluator was taught to pass the role-aware sidecar.  It only
    reads selected latent indices from answer JSONL and never reads source
    context, making the join safe for the answer-time contract.
    """
    if role_jsonl is None:
        return records
    role_rows = _load_role_sidecar(role_jsonl)
    enriched: list[dict[str, Any]] = []
    for record in records:
        record_id = str(record.get("id", ""))
        if record_id not in role_rows:
            raise ValueError(f"role sidecar is missing answer record {record_id}")
        sidecar = role_rows[record_id]
        answer_document = record.get("document_id", record_id)
        sidecar_document = sidecar["document_id"]
        if sidecar_document is not None and str(sidecar_document) != str(answer_document):
            raise ValueError(
                f"role sidecar for {record_id} points to document {sidecar_document}, "
                f"not {answer_document}"
            )
        answer_fingerprint = record.get("cache_fingerprint")
        sidecar_fingerprint = sidecar.get("cache_fingerprint")
        if (
            answer_fingerprint is not None
            and sidecar_fingerprint is not None
            and str(answer_fingerprint) != str(sidecar_fingerprint)
        ):
            raise ValueError(f"role sidecar for {record_id} uses an incompatible cache")
        selected = record.get("selected_latent_indices")
        if not isinstance(selected, list):
            raise ValueError(f"answer record {record_id} has no selected_latent_indices list")
        selected_set = {int(index) for index in selected}
        updated = dict(record)
        for role, metric in (
            ("answer", "answer_latent_recall"),
            ("all_answer", "all_answer_latent_recall"),
            ("key_answer", "key_answer_joint_recall"),
            ("support", "support_latent_recall"),
        ):
            indices = {int(index) for index in sidecar["roles"].get(role, [])}
            updated[metric] = (
                len(selected_set & indices) / len(indices) if indices else None
            )
        updated["role_labels_available"] = sidecar.get("role_labels_available")
        enriched.append(updated)
    return enriched


def _mean_or_none(records: list[dict[str, Any]], field: str) -> float | None:
    """Average an optional diagnostic without turning missing labels into zero."""
    values = [float(record[field]) for record in records if record.get(field) is not None]
    return statistics.mean(values) if values else None


def _valid_count(records: list[dict[str, Any]], field: str) -> int:
    return sum(record.get(field) is not None for record in records)


def _percentile(records: list[dict[str, Any]], field: str, percentile: float) -> float:
    """Return an interpolated percentile for one recorded metric.

    Keeping this calculation here avoids a NumPy dependency in the evaluation
    summary and makes the cold-start latency protocol reproducible across
    machines.
    """
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    values = sorted(
        float(record[field])
        for record in records
        if record.get(field) is not None
    )
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def _max(records: list[dict[str, Any]], field: str) -> int:
    values = [int(record[field]) for record in records if record.get(field) is not None]
    return max(values) if values else 0


def _budget_sweep_summary(records: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    """Aggregate dynamic-K predicted coverage by discrete budget ratio."""
    buckets: dict[float, list[dict[str, float | int]]] = {}
    for record in records:
        for point in record.get("dynamic_budget_sweep", []) or []:
            ratio = float(point["ratio"])
            buckets.setdefault(ratio, []).append(point)
    output = []
    for ratio, points in sorted(buckets.items()):
        row: dict[str, float | int] = {
            "ratio": ratio,
            "k_mean": statistics.mean(float(point["k"]) for point in points),
            "predicted_coverage_mean": statistics.mean(
                float(point["predicted_coverage"]) for point in points
            ),
            "examples": len(points),
        }
        # Newer dynamic selectors may retain an optional post-rerank coverage
        # value per sweep point.  Keep this aggregation backward compatible
        # with older answer JSONL files that only have predicted coverage.
        selected = [point.get("selected_coverage") for point in points]
        selected = [float(value) for value in selected if value is not None]
        if selected:
            row["selected_coverage_mean"] = statistics.mean(selected)
        output.append(row)
    return output


_SOURCE_SPAN_FIELDS = (
    "selected_source_start_min",
    "selected_source_end_max",
    "selected_source_span_mean",
    "selected_source_token_extent",
    "selected_source_run_count",
)


def _metric_summary(records: list[dict[str, Any]], field: str) -> dict[str, float | int]:
    """Return stable mean/P50/P95/count statistics for one optional metric."""
    values = [record.get(field) for record in records if record.get(field) is not None]
    numeric = [float(value) for value in values]
    return {
        "mean": statistics.mean(numeric) if numeric else 0.0,
        "p50": _percentile(records, field, 50.0),
        "p95": _percentile(records, field, 95.0),
        "valid_count": len(numeric),
    }


def _source_span_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    return {field: _metric_summary(records, field) for field in _SOURCE_SPAN_FIELDS}


def _build_summary(
    args: argparse.Namespace,
    manifest: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate accuracy, latency, reuse, and memory under one protocol."""
    per_task: dict[str, list[float]] = {}
    for record in records:
        per_task.setdefault(record["task"], []).append(
            _ruler_score(
                record.get("answers", []),
                record["response"],
                record.get("scoring_function", "ruler_string_match_all"),
            )
        )
    scores = {task: statistics.mean(task_scores) for task, task_scores in per_task.items()}
    per_task_records = {
        task: [record for record in records if record["task"] == task]
        for task in per_task
    }
    per_task_budget = {
        task: {
            "k_mean": _mean(task_records, "k_decoder"),
            "k_p50": _percentile(task_records, "k_decoder", 50.0),
            "k_p95": _percentile(task_records, "k_decoder", 95.0),
            "selection_ratio_mean": _mean(task_records, "selection_ratio"),
            "predicted_coverage_mean": _mean(task_records, "coverage_predicted"),
            "selected_coverage_mean": _mean(task_records, "coverage_selected"),
        }
        for task, task_records in per_task_records.items()
    }
    per_task_source_span = {
        task: _source_span_summary(task_records)
        for task, task_records in per_task_records.items()
    }
    global_source_span = _source_span_summary(records)
    flat_source_span = {
        f"{field}_{statistic}": value
        for field, summary in global_source_span.items()
        for statistic, value in summary.items()
    }
    per_task_answer_nll = {
        task: {
            "full_mean": _mean(task_records, "answer_nll_full"),
            "selected_mean": _mean(task_records, "answer_nll_selected"),
            "null_mean": _mean(task_records, "answer_nll_null"),
            "delta_vs_full_mean": _mean(task_records, "answer_nll_delta_vs_full"),
            "retention_mean": _mean(task_records, "answer_nll_retention"),
            "valid_retention_count": sum(
                record.get("answer_nll_retention") is not None for record in task_records
            ),
        }
        for task, task_records in per_task_records.items()
        if any(bool(record.get("answer_nll_eval_only")) for record in task_records)
    }
    cache_miss_records = [record for record in records if not bool(record.get("cache_hit", False))]
    cold_start_records = [
        record for record in records
        if bool(record.get("cache_first_load", not record.get("cache_hit", False)))
    ]
    resident_records = [record for record in records if bool(record.get("cache_hit", False))]
    encoder_latency_total_ms = sum(float(row["encoder_latency_ms"]) for row in manifest)
    reuse_points = (1, 4, 16)
    role_summary = {
        f"{field}_mean": _mean_or_none(records, field)
        for field in (
            "answer_latent_recall",
            "all_answer_latent_recall",
            "key_answer_joint_recall",
            "support_latent_recall",
        )
    }
    role_summary.update({
        f"{field}_valid_count": _valid_count(records, field)
        for field in (
            "answer_latent_recall",
            "all_answer_latent_recall",
            "key_answer_joint_recall",
            "support_latent_recall",
        )
    })
    return {
        "summary_schema_version": 4,
        "checkpoint": args.checkpoint,
        "k_decoder": args.k_decoder,
        "selection_ratio": args.selection_ratio,
        "selector_mode": args.selector_mode,
        "oracle_role": getattr(args, "oracle_role", None),
        "role_sidecar": getattr(args, "role_jsonl", None),
        "position_policy": getattr(args, "position_policy", "compact"),
        "neighbor_radius": args.neighbor_radius,
        "coverage_rerank": getattr(args, "coverage_rerank", False),
        "coverage_weight": getattr(args, "coverage_weight", 0.25),
        "dynamic_k": getattr(args, "dynamic_k", False),
        "dynamic_coverage_threshold": getattr(args, "dynamic_coverage_threshold", None),
        "num_queries": len(records),
        "num_unique_contexts": len({str(record.get("document_id", record["id"])) for record in records}),
        "cache_hit_rate": _mean(records, "cache_hit"),
        "cache_cold_start_count": len(cold_start_records),
        "cache_miss_count": len(cache_miss_records),
        "cache_resident_count": len(resident_records),
        "repeat_each_query": args.repeat_each_query,
        "max_resident_caches": args.max_resident_caches,
        "reuse_queries": args.reuse_queries,
        "ruler_accuracy": statistics.mean(scores.values()) if scores else 0.0,
        "per_task_accuracy": scores,
        "per_task_budget": per_task_budget,
        "selected_source_span_summary": global_source_span,
        "per_task_source_span": per_task_source_span,
        **flat_source_span,
        "encoder_latency_ms_total": encoder_latency_total_ms,
        "encoder_latency_ms_amortized": encoder_latency_total_ms / args.reuse_queries,
        "encoder_latency_ms_amortized_by_reuse": {
            str(reuse): encoder_latency_total_ms / reuse for reuse in reuse_points
        },
        "cache_load_latency_ms_mean_all_queries": _mean(records, "cache_load_latency_ms"),
        "cache_load_latency_ms_mean": _mean(records, "cache_load_latency_ms"),
        "cache_cold_start_load_latency_ms_mean": _mean(
            cold_start_records, "cache_load_latency_ms"
        ),
        "cache_cold_start_load_latency_ms_p50": _percentile(
            cold_start_records, "cache_load_latency_ms", 50.0
        ),
        "cache_cold_start_load_latency_ms_p95": _percentile(
            cold_start_records, "cache_load_latency_ms", 95.0
        ),
        "cache_cold_start_load_latency_ms_max": _max(
            cold_start_records, "cache_load_latency_ms"
        ),
        "cache_miss_load_latency_ms_mean": _mean(
            cache_miss_records, "cache_load_latency_ms"
        ),
        "cache_miss_load_latency_ms_p50": _percentile(
            cache_miss_records, "cache_load_latency_ms", 50.0
        ),
        "cache_miss_load_latency_ms_p95": _percentile(
            cache_miss_records, "cache_load_latency_ms", 95.0
        ),
        # This is the estimated resident path for every query, including
        # misses after subtracting their measured disk load.
        "cache_resident_ttft_ms_mean": _mean(records, "cache_resident_ttft_ms"),
        "cache_resident_ttft_ms_p50": _percentile(records, "cache_resident_ttft_ms", 50.0),
        "cache_resident_ttft_ms_p95": _percentile(records, "cache_resident_ttft_ms", 95.0),
        # These are actual LRU hits and therefore an independently measurable
        # resident-cache protocol rather than a subtraction estimate.
        "cache_hit_ttft_ms_mean": _mean(resident_records, "ttft_ms"),
        "cache_hit_ttft_ms_p50": _percentile(resident_records, "ttft_ms", 50.0),
        "cache_hit_ttft_ms_p95": _percentile(resident_records, "ttft_ms", 95.0),
        "latency_protocol": {
            "ttft_ms": "cache_load + question_tokenize + selector + adapter + decoder_input + prefill",
            "cache_resident_ttft_ms": "ttft_ms - cache_load_latency_ms",
            "cold_start_load_latency_ms": "cache_first_load=true records only",
            "cache_miss_load_latency_ms": "cache_hit=false records only",
            "cache_hit_ttft_ms": "cache_hit=true records only",
        },
        "question_tokenize_latency_ms_mean": _mean(records, "question_tokenize_latency_ms"),
        "selector_latency_ms_mean": _mean(records, "selector_latency_ms"),
        "selector_peak_memory_bytes_max": _max(records, "selector_peak_memory_bytes"),
        "adapter_latency_ms_mean": _mean(records, "adapter_latency_ms"),
        "adapter_peak_memory_bytes_max": _max(records, "adapter_peak_memory_bytes"),
        "decoder_input_latency_ms_mean": _mean(records, "decoder_input_latency_ms"),
        "decoder_prefill_latency_ms_mean": _mean(records, "decoder_prefill_latency_ms"),
        "decoder_decode_latency_ms_mean": _mean(records, "decoder_decode_latency_ms"),
        "decoder_peak_memory_bytes_max": _max(records, "decoder_peak_memory_bytes"),
        "ttft_ms_mean": _mean(records, "ttft_ms"),
        "total_query_latency_ms_mean": _mean(records, "total_query_latency_ms"),
        "decoder_kv_cache_bytes_estimate_mean": _mean(
            records, "decoder_kv_cache_bytes_estimate"
        ),
        "selection_ratio_actual_mean": _mean(records, "selection_ratio"),
        "selection_ratio_actual_p50": _percentile(records, "selection_ratio", 50.0),
        "selection_ratio_actual_p95": _percentile(records, "selection_ratio", 95.0),
        "k_decoder_mean": _mean(records, "k_decoder"),
        "k_decoder_p50": _percentile(records, "k_decoder", 50.0),
        "k_decoder_p95": _percentile(records, "k_decoder", 95.0),
        "predicted_coverage_mean": _mean(records, "coverage_predicted"),
        "selected_coverage_mean": _mean(records, "coverage_selected"),
        "dynamic_budget_sweep": _budget_sweep_summary(records),
        "oracle_latent_recall_mean": _mean(records, "oracle_latent_recall"),
        **role_summary,
        "answer_nll_eval_enabled": any(bool(record.get("answer_nll_eval_only")) for record in records),
        "answer_nll_valid_retention_count": sum(
            record.get("answer_nll_retention") is not None for record in records
        ),
        "answer_nll_full_mean": _mean(records, "answer_nll_full"),
        "answer_nll_selected_mean": _mean(records, "answer_nll_selected"),
        "answer_nll_null_mean": _mean(records, "answer_nll_null"),
        "answer_nll_delta_vs_full_mean": _mean(records, "answer_nll_delta_vs_full"),
        "answer_nll_retention_mean": _mean(records, "answer_nll_retention"),
        "answer_nll_retention_clipped_mean": _mean(records, "answer_nll_retention_clipped"),
        "answer_likelihood_ratio_vs_full_mean": _mean(
            records, "answer_likelihood_ratio_vs_full"
        ),
        "answer_nll_eval_latency_ms_mean": _mean(records, "answer_nll_eval_latency_ms"),
        "per_task_answer_nll": per_task_answer_nll,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate query-aware LCLM on prepared RULER NIAH")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-dir", required=True, help="directory containing memory.jsonl and queries.jsonl")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--cache-dir",
                        help="reuse an existing latent cache instead of rebuilding it")
    budget_group = parser.add_mutually_exclusive_group(required=True)
    budget_group.add_argument("--k-decoder", type=int)
    budget_group.add_argument("--selection-ratio", type=float)
    parser.add_argument("--selector-mode", default="cosine",
                        choices=[
                            "full", "cosine", "learned", "random", "uniform", "oracle",
                            "utility_oracle",
                        ])
    parser.add_argument("--selector-checkpoint")
    parser.add_argument("--oracle-jsonl")
    parser.add_argument(
        "--role-jsonl",
        help="optional role-aware oracle sidecar for offline recall diagnostics",
    )
    parser.add_argument(
        "--oracle-role",
        help="optional role or comma-separated role union for oracle selection",
    )
    parser.add_argument("--utility-oracle-jsonl")
    parser.add_argument("--random-seed", type=int, default=4231)
    parser.add_argument("--neighbor-radius", type=int, default=0,
                        help="fixed-budget source-neighbour expansion for cosine/learned modes")
    parser.add_argument("--coverage-rerank", action="store_true",
                        help="re-rank relevance scores with structural source-group coverage")
    parser.add_argument("--coverage-weight", type=float, default=0.25)
    parser.add_argument("--dynamic-k", action="store_true",
                        help="choose the smallest discrete budget meeting predicted coverage")
    parser.add_argument("--dynamic-coverage-threshold", type=float, default=0.75)
    parser.add_argument("--dynamic-budget-ratios", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--slot-policy", default="none",
                        choices=["none", "entity", "sentence", "content_tokens"],
                        help="decompose the question into multi-slot sub-queries")
    parser.add_argument("--slot-max-slots", type=int, default=16)
    parser.add_argument("--slot-combine", default="union",
                        choices=["union", "rank_fusion", "max"])
    parser.add_argument("--slot-fusion-k", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--reuse-queries", type=int, default=1,
                        help="report amortization assuming this many queries per cached context")
    parser.add_argument("--repeat-each-query", type=int, default=1,
                        help="execute adjacent query replicas to measure real cache-hit reuse")
    parser.add_argument("--max-resident-caches", type=int, default=1)
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--position-policy", choices=["compact", "full_layout"],
                        default="full_layout")
    parser.add_argument("--compute-answer-nll", action="store_true")
    parser.add_argument("--max-examples", type=int, default=0,
                        help="0 evaluates every query")
    args = parser.parse_args()
    if args.reuse_queries <= 0:
        raise ValueError("--reuse-queries must be positive")
    try:
        dynamic_budget_ratios = tuple(
            float(value) for value in args.dynamic_budget_ratios.split(",") if value.strip()
        )
    except ValueError as exc:
        raise ValueError("--dynamic-budget-ratios must be comma-separated numbers") from exc

    input_dir = Path(args.input_dir)
    work_dir = Path(args.work_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else work_dir / "cache"
    answers_path = work_dir / "answers.jsonl"
    work_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_cache and not args.cache_dir:
        _run([
            sys.executable, "-m", "inference.query_aware", "cache",
            "--checkpoint", args.checkpoint,
            "--memory-jsonl", str(input_dir / "memory.jsonl"),
            "--cache-dir", str(cache_dir),
            "--device", args.device,
            "--dtype", args.dtype,
        ])
    answer_command = [
        sys.executable, "-m", "inference.query_aware", "answer",
        "--checkpoint", args.checkpoint,
        "--cache-dir", str(cache_dir),
        "--queries-jsonl", str(input_dir / "queries.jsonl"),
        "--out", str(answers_path),
        "--max-tokens", str(args.max_tokens),
        "--device", args.device,
        "--dtype", args.dtype,
        "--selector-mode", args.selector_mode,
        "--random-seed", str(args.random_seed),
        "--repeat-each-query", str(args.repeat_each_query),
        "--max-resident-caches", str(args.max_resident_caches),
        "--position-policy", args.position_policy,
        "--max-examples", str(args.max_examples),
    ]
    if args.k_decoder is not None:
        answer_command.extend(["--k-decoder", str(args.k_decoder)])
    else:
        answer_command.extend(["--selection-ratio", str(args.selection_ratio)])
    if args.selector_checkpoint:
        answer_command.extend(["--selector-checkpoint", args.selector_checkpoint])
    # The role-aware sidecar also carries the historical answer oracle fields,
    # so it can serve as the sole diagnostic sidecar for learned evaluations.
    diagnostic_oracle = args.role_jsonl or args.oracle_jsonl
    if diagnostic_oracle:
        answer_command.extend(["--oracle-jsonl", diagnostic_oracle])
    if args.oracle_role:
        answer_command.extend(["--oracle-role", args.oracle_role])
    if args.utility_oracle_jsonl:
        answer_command.extend(["--utility-oracle-jsonl", args.utility_oracle_jsonl])
    if args.compute_answer_nll:
        answer_command.append("--compute-answer-nll")
    if args.neighbor_radius:
        answer_command.extend(["--neighbor-radius", str(args.neighbor_radius)])
    if args.coverage_rerank:
        answer_command.append("--coverage-rerank")
    answer_command.extend(["--coverage-weight", str(args.coverage_weight)])
    if args.dynamic_k:
        answer_command.extend([
            "--dynamic-k",
            "--dynamic-coverage-threshold", str(args.dynamic_coverage_threshold),
            "--dynamic-budget-ratios", ",".join(str(value) for value in dynamic_budget_ratios),
        ])
    if args.slot_policy != "none":
        answer_command.extend([
            "--slot-policy", args.slot_policy,
            "--slot-max-slots", str(args.slot_max_slots),
            "--slot-combine", args.slot_combine,
            "--slot-fusion-k", str(args.slot_fusion_k),
        ])
    _run(answer_command)

    manifest = json.loads((cache_dir / "manifest.json").read_text())
    records = [json.loads(line) for line in answers_path.read_text().splitlines() if line.strip()]
    # Older answer runs may have been generated with the basic oracle sidecar;
    # enrich them in-process when a role sidecar is explicitly supplied.
    records = _attach_role_metrics(records, args.role_jsonl)
    evaluated_documents = {str(record.get("document_id", record["id"])) for record in records}
    manifest = [row for row in manifest if str(row.get("document_id", row["id"])) in evaluated_documents]
    summary = _build_summary(args, manifest, records)
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
