"""Offline role-label oracle-K curve for a trained role-aware selector.

The script evaluates hard selections at several budgets using only cached
latents and short questions, then uses offline numbered role groups to report
the smallest budget reaching a joint-coverage target.  It is an oracle
diagnostic, not an answer-time policy and not a global NLL subset optimum.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch

from latent_context import QuestionEncoder, from_pretrained
from latent_context.role_selector import load_role_selector_checkpoint
from latent_context.selector import load_latent_memory


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _fraction(selected: set[int], values: list[int]) -> float | None:
    target = {int(value) for value in values}
    return len(selected & target) / len(target) if target else None


def _groups_fraction(selected: set[int], roles: dict[str, list[int]], prefix: str) -> float | None:
    groups = [values for name, values in roles.items() if name.startswith(prefix + "_")]
    values = [_fraction(selected, group) for group in groups if group]
    return sum(value > 0 for value in values) / len(values) if values else None


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    model, _, _ = from_pretrained(args.checkpoint, device=args.device, dtype=args.dtype)
    model.eval()
    selector, metadata = load_role_selector_checkpoint(
        args.selector_checkpoint, model.memory_fingerprint()
    )
    selector = selector.to(args.device).eval()
    question_encoder = QuestionEncoder(
        model.encoder.embed_model,
        model.encoder.embed_model.config.hidden_size,
        freeze_backbone=True,
    ).to(args.device).eval()
    tokenizer = model.encoder.embed_tokenizer
    queries = _read(Path(args.queries_jsonl))
    roles = {str(row["id"]): row for row in _read(Path(args.role_jsonl))}
    ratios = tuple(sorted({float(value) for value in args.budget_ratios.split(",") if value.strip()}))
    rows: list[dict[str, Any]] = []
    for query in queries:
        query_id = str(query["id"])
        sidecar = roles.get(query_id)
        if sidecar is None:
            raise ValueError(f"missing role row {query_id}")
        memory = load_latent_memory(Path(args.cache_dir) / str(query.get("document_id", query_id)), device=args.device)
        encoded = tokenizer([str(query.get("selector_query", query.get("question")))], padding=True, return_tensors="pt", add_special_tokens=False)
        question = question_encoder(encoded["input_ids"].to(args.device), encoded["attention_mask"].to(args.device))
        role_mapping = sidecar.get("role_latent_indices", {})
        points: list[dict[str, Any]] = []
        for ratio in ratios:
            k = max(1, min(memory.num_latents, math.ceil(ratio * memory.num_latents)))
            selected = set(int(value) for value in selector(question, memory, k).indices[0].tolist())
            key_fraction = _groups_fraction(selected, role_mapping, "key_answer")
            support_fraction = _groups_fraction(selected, role_mapping, "support")
            active = [value for value in (key_fraction, support_fraction) if value is not None]
            joint = min(active) if active else None
            points.append({
                "ratio": ratio,
                "k": k,
                "answer_recall": _fraction(selected, role_mapping.get("all_answer", role_mapping.get("answer", []))),
                "key_answer_recall": _fraction(selected, role_mapping.get("key_answer", [])),
                "support_recall": _fraction(selected, role_mapping.get("support", [])),
                "key_answer_group_fraction": key_fraction,
                "support_group_fraction": support_fraction,
                "joint_role_recall": joint,
            })
        eligible = [point for point in points if point["joint_role_recall"] is not None and point["joint_role_recall"] >= args.joint_threshold]
        rows.append({
            "id": query_id,
            "task": query.get("task"),
            "budget_curve": points,
            "oracle_min_ratio": eligible[0]["ratio"] if eligible else None,
            "oracle_min_k": eligible[0]["k"] if eligible else None,
        })

    curve = []
    for index, ratio in enumerate(ratios):
        points = [row["budget_curve"][index] for row in rows]
        joints = [point["joint_role_recall"] for point in points if point["joint_role_recall"] is not None]
        curve.append({
            "ratio": ratio,
            "examples": len(points),
            "joint_role_recall_mean": statistics.mean(joints) if joints else None,
            "joint_complete_fraction": sum(value >= args.joint_threshold for value in joints) / len(joints) if joints else None,
            "answer_recall_mean": statistics.mean(point["answer_recall"] for point in points if point["answer_recall"] is not None),
        })
    minima = [row["oracle_min_ratio"] for row in rows if row["oracle_min_ratio"] is not None]
    return {
        "schema_version": 1,
        "oracle_scope": "minimum declared budget for offline joint-role coverage; not global NLL subset optimum",
        "joint_threshold": args.joint_threshold,
        "budget_ratios": ratios,
        "selector_checkpoint": args.selector_checkpoint,
        "selector_metadata": metadata,
        "curve": curve,
        "oracle_min_ratio_distribution": {
            str(ratio): minima.count(ratio) for ratio in ratios
        },
        "rows": rows,
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
    parser.add_argument("--joint-threshold", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = parser.parse_args()
    report = run(args)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"out": str(output), "rows": len(report["rows"])}))


if __name__ == "__main__":
    main()
