"""Calibrate role-score dynamic-K on independent validation documents only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

import torch

from latent_context import QuestionEncoder, from_pretrained
from latent_context.role_dynamic_k import role_mass_budget_sweep
from latent_context.role_selector import load_role_selector_checkpoint
from latent_context.selector import load_latent_memory


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _fraction(selected: set[int], values: Any) -> float | None:
    target = {int(value) for value in values} if isinstance(values, list) else set()
    return len(selected & target) / len(target) if target else None


def _group_fraction(selected: set[int], roles: dict[str, Any], prefix: str) -> float | None:
    groups = [
        {int(value) for value in values}
        for name, values in roles.items()
        if name.startswith(prefix + "_")
        and name[len(prefix) + 1:].isascii()
        and name[len(prefix) + 1:].isdecimal()
        and isinstance(values, list)
        and values
    ]
    return sum(bool(selected & group) for group in groups) / len(groups) if groups else None


def _true_metrics(selected: set[int], roles: dict[str, Any]) -> dict[str, float | None]:
    key_group = _group_fraction(selected, roles, "key_answer")
    support_group = _group_fraction(selected, roles, "support")
    active = [value for value in (key_group, support_group) if value is not None]
    return {
        "answer_recall": _fraction(selected, roles.get("all_answer", roles.get("answer", []))),
        "key_answer_recall": _fraction(selected, roles.get("key_answer", [])),
        "support_recall": _fraction(selected, roles.get("support", [])),
        "joint_group_recall": min(active) if active else None,
    }


def _is_sufficient(metrics: dict[str, float | None], targets: dict[str, float]) -> bool:
    return all(metrics.get(name) is not None and float(metrics[name]) >= target for name, target in targets.items())


def _mean(rows: list[dict[str, Any]], name: str) -> float:
    values = [float(row[name]) for row in rows if row.get(name) is not None]
    return statistics.mean(values) if values else 0.0


def _policy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "examples": len(rows),
        "mean_ratio": _mean(rows, "ratio"),
        "success_rate": _mean(rows, "sufficient"),
        "answer_recall_mean": _mean(rows, "answer_recall"),
        "key_answer_recall_mean": _mean(rows, "key_answer_recall"),
        "support_recall_mean": _mean(rows, "support_recall"),
        "joint_group_recall_mean": _mean(rows, "joint_group_recall"),
        "budget_distribution": {
            str(ratio): sum(float(row["ratio"]) == ratio for row in rows)
            for ratio in sorted({float(row["ratio"]) for row in rows})
        },
    }


def _chosen_rows(
    rows: list[dict[str, Any]], temperature: float, threshold: float,
) -> list[dict[str, Any]]:
    output = []
    key = str(temperature)
    for row in rows:
        points = row["temperature_sweeps"][key]
        chosen = next(
            (point for point in points if float(point["predicted_coverage"]) >= threshold),
            points[-1],
        )
        output.append({
            "id": row["id"],
            "length_tag": row["length_tag"],
            "answer_count": row["answer_count"],
            "ratio": float(chosen["ratio"]),
            "predicted_coverage": float(chosen["predicted_coverage"]),
            "sufficient": bool(chosen["sufficient"]),
            **chosen["true_metrics"],
        })
    return output


def _candidate_thresholds(rows: list[dict[str, Any]], temperature: float) -> list[float]:
    key = str(temperature)
    values = sorted({
        float(point["predicted_coverage"])
        for row in rows for point in row["temperature_sweeps"][key][:-1]
    })
    thresholds = {0.0, 1.0}
    thresholds.update(min(1.0, math.nextafter(value, math.inf)) for value in values)
    return sorted(thresholds)


@torch.inference_mode()
def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    queries_path = Path(args.queries_jsonl)
    roles_path = Path(args.role_jsonl)
    selector_path = Path(args.selector_checkpoint)
    queries = _read(queries_path)
    roles = {str(row["id"]): row for row in _read(roles_path)}
    if not queries or any(row.get("split") != "validation" for row in queries):
        raise ValueError("dynamic-K calibration requires an explicit validation-only query set")
    if set(str(row["id"]) for row in queries) != set(roles):
        raise ValueError("validation query and role sidecar IDs differ")

    model, _, _ = from_pretrained(args.checkpoint, device=args.device, dtype=args.dtype)
    model.eval()
    fingerprint = model.memory_fingerprint()
    selector, selector_metadata = load_role_selector_checkpoint(selector_path, fingerprint)
    selector = selector.to(args.device).eval()
    question_encoder = QuestionEncoder(
        model.encoder.embed_model,
        model.encoder.embed_model.config.hidden_size,
        freeze_backbone=True,
    ).to(args.device).eval()
    tokenizer = model.encoder.embed_tokenizer
    ratios = tuple(sorted({float(value) for value in args.budget_ratios.split(",") if value}))
    temperatures = tuple(float(value) for value in args.temperatures.split(",") if value)
    targets = {
        "answer_recall": args.answer_recall_target,
        "key_answer_recall": args.key_answer_recall_target,
        "support_recall": args.support_recall_target,
        "joint_group_recall": args.joint_group_recall_target,
    }
    rows: list[dict[str, Any]] = []
    for query_index, query in enumerate(queries, start=1):
        query_id = str(query["id"])
        role_mapping = roles[query_id].get("role_latent_indices", {})
        memory_id = str(query.get("document_id", query_id))
        memory = load_latent_memory(Path(args.cache_dir) / memory_id, device=args.device)
        if memory.fingerprint != fingerprint:
            raise ValueError(f"validation cache {memory_id} has incompatible fingerprint")
        encoded = tokenizer(
            [str(query.get("selector_query", query.get("question")))],
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        question = question_encoder(
            encoded["input_ids"].to(args.device), encoded["attention_mask"].to(args.device)
        )
        temperature_sweeps: dict[str, list[dict[str, Any]]] = {}
        for temperature in temperatures:
            sweep, selections, _ = role_mass_budget_sweep(
                selector,
                question,
                memory,
                budget_ratios=ratios,
                temperature=temperature,
            )
            points = []
            for point in sweep:
                selected = set(int(value) for value in selections[float(point["ratio"])].tolist())
                true_metrics = _true_metrics(selected, role_mapping)
                points.append({
                    **point,
                    "true_metrics": true_metrics,
                    "sufficient": _is_sufficient(true_metrics, targets),
                })
            temperature_sweeps[str(temperature)] = points
        rows.append({
            "id": query_id,
            "document_id": memory_id,
            "length_tag": query.get("length_tag"),
            "answer_count": len(query.get("answers", [])),
            "temperature_sweeps": temperature_sweeps,
        })
        if query_index % 16 == 0:
            print(json.dumps({"scored": query_index, "total": len(queries)}), flush=True)

    # The true fixed-50 baseline is independent of score temperature.
    fixed_index = min(range(len(ratios)), key=lambda index: abs(ratios[index] - 0.5))
    fixed_rows = []
    first_temperature = str(temperatures[0])
    for row in rows:
        point = row["temperature_sweeps"][first_temperature][fixed_index]
        fixed_rows.append({
            "ratio": float(point["ratio"]),
            "sufficient": bool(point["sufficient"]),
            **point["true_metrics"],
        })
    fixed_summary = _policy_summary(fixed_rows)

    candidates = []
    for temperature in temperatures:
        for threshold in _candidate_thresholds(rows, temperature):
            summary = _policy_summary(_chosen_rows(rows, temperature, threshold))
            role_constraints = all(
                float(summary[name]) + args.metric_tolerance >= float(fixed_summary[name])
                for name in (
                    "success_rate",
                    "answer_recall_mean",
                    "key_answer_recall_mean",
                    "support_recall_mean",
                    "joint_group_recall_mean",
                )
            )
            candidates.append({
                "temperature": temperature,
                "threshold": threshold,
                "meets_fixed_50_role_constraints": role_constraints,
                **summary,
            })
    feasible = [row for row in candidates if row["meets_fixed_50_role_constraints"]]
    pool = feasible or candidates
    selected_candidate = min(
        pool,
        key=lambda row: (
            float(row["mean_ratio"]),
            -float(row["success_rate"]),
            -float(row["answer_recall_mean"]),
            float(row["temperature"]),
            float(row["threshold"]),
        ),
    )
    selected_rows = _chosen_rows(
        rows,
        float(selected_candidate["temperature"]),
        float(selected_candidate["threshold"]),
    )
    qualified = bool(
        selected_candidate["meets_fixed_50_role_constraints"]
        and float(selected_candidate["mean_ratio"]) < 0.5
    )
    return {
        "schema_version": 1,
        "policy": "role_score_mass_v1",
        "scope": "validation-only calibration; no RULER test labels or NLL used",
        "base_memory_fingerprint": fingerprint,
        "selector_checkpoint": str(selector_path),
        "selector_checkpoint_sha256": hashlib.sha256(
            (selector_path / "role_selector.safetensors").read_bytes()
        ).hexdigest(),
        "selector_metadata": selector_metadata,
        "validation_queries_sha256": hashlib.sha256(queries_path.read_bytes()).hexdigest(),
        "validation_roles_sha256": hashlib.sha256(roles_path.read_bytes()).hexdigest(),
        "validation_examples": len(rows),
        "budget_ratios": ratios,
        "coverage_targets": targets,
        "metric_tolerance": args.metric_tolerance,
        "temperature": selected_candidate["temperature"],
        "coverage_threshold": selected_candidate["threshold"],
        "qualified_against_fixed_50": qualified,
        "fixed_50_validation": fixed_summary,
        "selected_policy_validation": {
            key: value for key, value in selected_candidate.items()
            if key not in {"temperature", "threshold"}
        },
        "selected_rows": selected_rows,
        "candidate_count": len(candidates),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--queries-jsonl", required=True)
    parser.add_argument("--role-jsonl", required=True)
    parser.add_argument("--selector-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--budget-ratios", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--temperatures", default="0.02,0.05,0.1,0.2,0.5,1.0")
    parser.add_argument("--answer-recall-target", type=float, default=0.90)
    parser.add_argument("--key-answer-recall-target", type=float, default=0.90)
    parser.add_argument("--support-recall-target", type=float, default=0.75)
    parser.add_argument("--joint-group-recall-target", type=float, default=1.0)
    parser.add_argument("--metric-tolerance", type=float, default=0.005)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = parser.parse_args()
    if args.metric_tolerance < 0:
        raise ValueError("metric tolerance must be non-negative")
    report = calibrate(args)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "out": str(output),
        "temperature": report["temperature"],
        "threshold": report["coverage_threshold"],
        "qualified": report["qualified_against_fixed_50"],
    }))


if __name__ == "__main__":
    main()
