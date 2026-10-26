"""Evaluate confidence-based per-query dynamic-K with RoleAwareSelector."""

from __future__ import annotations

import argparse
import hashlib
import json
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn

import inference.query_aware as query_aware
from inference.examples.eval_query_aware_niah import _attach_role_metrics, _build_summary
from latent_context.role_dynamic_k import select_calibrated_role_budget
from latent_context.role_selector import load_role_selector_checkpoint
from latent_context.selector import DynamicSelectionOutput


class _RecordingRoleSelector(nn.Module):
    """Expose the current query/cache to the injected dynamic policy."""

    def __init__(self, selector: nn.Module) -> None:
        super().__init__()
        self.selector = selector
        self.last_question: torch.Tensor | None = None
        self.last_memory = None

    def forward(self, question: torch.Tensor, memory, k: int):
        self.last_question = question
        self.last_memory = memory
        return self.selector(question, memory, k)


def confidence_budget(
    scores: torch.Tensor,
    memory_length: int | None = None,
    *,
    minimum_ratio: float = 0.25,
    budget_ratios: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    coverage_threshold: float = 0.90,
    temperature: float = 1.0,
    **_: object,
) -> DynamicSelectionOutput:
    """Choose the smallest hard Top-K whose query score mass reaches threshold."""
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a non-empty [M] tensor")
    if memory_length is not None and int(memory_length) != scores.numel():
        raise ValueError("memory_length does not match scores")
    if not 0.0 < minimum_ratio <= 1.0:
        raise ValueError("minimum_ratio must be in (0, 1]")
    if not 0.0 <= coverage_threshold <= 1.0:
        raise ValueError("coverage threshold must be in [0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    ratios = sorted({float(value) for value in budget_ratios if float(value) >= minimum_ratio})
    if not ratios:
        ratios = [1.0]
    ranked = torch.argsort(scores, descending=True, stable=True)
    probability = torch.softmax(scores.float() / temperature, dim=0)
    cumulative = torch.cumsum(probability.index_select(0, ranked), dim=0)
    sweep: list[dict[str, float | int]] = []
    chosen: tuple[float, int, float] | None = None
    for ratio in ratios:
        k = max(1, min(int(scores.numel()), int(torch.ceil(torch.tensor(ratio * scores.numel())).item())))
        mass = float(cumulative[k - 1].item())
        sweep.append({"ratio": ratio, "k": k, "predicted_coverage": mass})
        if chosen is None and mass >= coverage_threshold:
            chosen = (ratio, k, mass)
    if chosen is None:
        point = sweep[-1]
        chosen = (float(point["ratio"]), int(point["k"]), float(point["predicted_coverage"]))
    ratio, k, mass = chosen
    indices = torch.sort(ranked[:k]).values
    return DynamicSelectionOutput(
        indices=indices,
        scores=scores,
        selected_count=k,
        selected_ratio=ratio,
        predicted_coverage=mass,
        budget_sweep=sweep,
        selected_coverage=mass,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--queries-jsonl", required=True)
    parser.add_argument("--role-jsonl", required=True)
    parser.add_argument("--selector-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--selection-ratio", type=float, default=0.25)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--dynamic-budget-ratios", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--position-policy", default="full_layout", choices=("compact", "full_layout"))
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--random-seed", type=int, default=6201)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--compute-answer-nll", action="store_true")
    args = parser.parse_args()
    ratios = tuple(float(value) for value in args.dynamic_budget_ratios.split(",") if value.strip())
    calibration_path = Path(args.calibration)
    calibration = json.loads(calibration_path.read_text())
    if calibration.get("schema_version") != 1 or calibration.get("policy") != "role_score_mass_v1":
        raise ValueError("unsupported dynamic-K calibration")
    calibrated_ratios = tuple(float(value) for value in calibration.get("budget_ratios", []))
    if ratios != calibrated_ratios:
        raise ValueError("evaluation budget ratios differ from calibration")
    threshold = float(calibration["coverage_threshold"])
    temperature = float(calibration["temperature"])
    qualified = bool(calibration.get("qualified_against_fixed_50"))
    fallback_ratio = None if qualified else 0.5

    original_loader = query_aware.load_selector_checkpoint
    original_dynamic = query_aware.select_dynamic_budget
    active_selector: _RecordingRoleSelector | None = None

    def load_role_selector(_path: str, expected_fingerprint: str):
        nonlocal active_selector
        selector, metadata = load_role_selector_checkpoint(
            args.selector_checkpoint, expected_fingerprint
        )
        expected_sha = calibration.get("selector_checkpoint_sha256")
        actual_sha = hashlib.sha256(
            (Path(args.selector_checkpoint) / "role_selector.safetensors").read_bytes()
        ).hexdigest()
        if expected_sha != actual_sha:
            raise ValueError("dynamic-K calibration belongs to a different selector checkpoint")
        if calibration.get("base_memory_fingerprint") != expected_fingerprint:
            raise ValueError("dynamic-K calibration belongs to a different latent encoder")
        active_selector = _RecordingRoleSelector(selector)
        return active_selector, None, metadata

    def calibrated_budget(scores: torch.Tensor, memory_length: int | None = None, **_: object):
        if active_selector is None or active_selector.last_question is None or active_selector.last_memory is None:
            raise RuntimeError("role-aware selector state is unavailable for dynamic selection")
        if memory_length is not None and int(memory_length) != int(scores.numel()):
            raise ValueError("memory length does not match selector scores")
        return select_calibrated_role_budget(
            active_selector.selector,
            active_selector.last_question,
            active_selector.last_memory,
            minimum_ratio=args.selection_ratio,
            budget_ratios=ratios,
            coverage_threshold=threshold,
            temperature=temperature,
            forced_ratio=fallback_ratio,
        )

    query_aware.load_selector_checkpoint = load_role_selector
    query_aware.select_dynamic_budget = calibrated_budget
    try:
        query_aware.answer_queries(
            checkpoint=args.checkpoint,
            cache_dir=args.cache_dir,
            queries_jsonl=args.queries_jsonl,
            out=args.out,
            k_decoder=None,
            selection_ratio=args.selection_ratio,
            max_tokens=args.max_tokens,
            device=args.device,
            dtype=args.dtype,
            selector_mode="learned",
            selector_checkpoint=args.selector_checkpoint,
            oracle_jsonl=args.role_jsonl,
            random_seed=args.random_seed,
            max_resident_caches=1,
            repeat_each_query=1,
            position_policy=args.position_policy,
            compute_answer_nll=args.compute_answer_nll,
            dynamic_k=True,
            dynamic_coverage_threshold=threshold,
            dynamic_budget_ratios=ratios,
        )
    finally:
        query_aware.load_selector_checkpoint = original_loader
        query_aware.select_dynamic_budget = original_dynamic

    output = Path(args.out)
    records = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    records = _attach_role_metrics(records, args.role_jsonl)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    manifest = json.loads((Path(args.cache_dir) / "manifest.json").read_text())
    document_ids = {str(row.get("document_id", row["id"])) for row in records}
    manifest = [
        row for row in manifest
        if str(row.get("document_id", row["id"])) in document_ids
    ]
    summary_args = Namespace(
        checkpoint=args.checkpoint,
        k_decoder=None,
        selection_ratio=args.selection_ratio,
        selector_mode="role_aware",
        oracle_role=None,
        role_jsonl=args.role_jsonl,
        position_policy=args.position_policy,
        neighbor_radius=0,
        coverage_rerank=False,
        coverage_weight=0.0,
        dynamic_k=True,
        dynamic_coverage_threshold=threshold,
        repeat_each_query=1,
        max_resident_caches=1,
        reuse_queries=1,
    )
    summary = _build_summary(summary_args, manifest, records)
    summary.update({
        "role_selector_checkpoint": args.selector_checkpoint,
        "dynamic_calibration": str(calibration_path),
        "dynamic_calibration_policy": calibration["policy"],
        "dynamic_calibration_temperature": temperature,
        "dynamic_calibration_qualified_against_fixed_50": qualified,
        "dynamic_calibration_fallback_ratio": fallback_ratio,
    })
    summary_path = output.parent / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "out": str(output),
        "summary": str(summary_path),
        "queries": len(records),
        "ruler_accuracy": summary["ruler_accuracy"],
        "mean_ratio": summary["selection_ratio_actual_mean"],
    }))


if __name__ == "__main__":
    main()
