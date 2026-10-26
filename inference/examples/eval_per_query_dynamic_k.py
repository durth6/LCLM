"""Run a confidence-based, genuinely per-query dynamic-K experiment.

The stable evaluator already contains the full latent-only answer path.  This
wrapper replaces only its budget decision with a query-specific cumulative
score-mass rule, leaving hard Top-K, source ordering, cache provenance, and
NLL accounting unchanged.  The replacement is injected in memory before the
evaluator is called; no oracle/NLL labels or source text are read online.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import inference.query_aware as query_aware
from latent_context.selector import DynamicSelectionOutput


def confidence_budget(
    scores: torch.Tensor,
    memory_length: int | None = None,
    *,
    minimum_ratio: float = 0.25,
    budget_ratios: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    confidence_threshold: float = 0.90,
    temperature: float = 1.0,
    **_: object,
) -> DynamicSelectionOutput:
    """Choose the smallest K whose query-specific relevance mass is enough."""
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a non-empty [M] tensor")
    if memory_length is not None and int(memory_length) != scores.numel():
        raise ValueError("memory_length does not match scores")
    if not 0.0 < minimum_ratio <= 1.0:
        raise ValueError("minimum_ratio must be in (0, 1]")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
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
        k = max(1, min(scores.numel(), int(torch.ceil(torch.tensor(ratio * scores.numel())).item())))
        mass = float(cumulative[k - 1].item())
        sweep.append({"ratio": ratio, "k": k, "predicted_coverage": mass})
        if chosen is None and mass >= confidence_threshold:
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
    parser.add_argument("--out", required=True)
    parser.add_argument("--selector-checkpoint", required=True)
    parser.add_argument("--oracle-jsonl")
    parser.add_argument("--role-jsonl")
    parser.add_argument("--selection-ratio", type=float, default=0.25)
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--confidence-temperature", type=float, default=1.0)
    parser.add_argument("--dynamic-budget-ratios", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--position-policy", default="full_layout", choices=("compact", "full_layout"))
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--random-seed", type=int, default=6201)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--reuse-queries", type=int, default=1)
    parser.add_argument("--repeat-each-query", type=int, default=1)
    parser.add_argument("--max-resident-caches", type=int, default=1)
    parser.add_argument("--compute-answer-nll", action="store_true")
    args = parser.parse_args()
    ratios = tuple(float(value) for value in args.dynamic_budget_ratios.split(",") if value.strip())

    # answer_queries resolves the function from inference.query_aware's module
    # namespace. Replacing it here keeps the established I/O and accounting
    # implementation while changing only the online per-query budget rule.
    query_aware.select_dynamic_budget = confidence_budget
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
        oracle_jsonl=args.role_jsonl or args.oracle_jsonl,
        random_seed=args.random_seed,
        max_resident_caches=args.max_resident_caches,
        repeat_each_query=args.repeat_each_query,
        position_policy=args.position_policy,
        compute_answer_nll=args.compute_answer_nll,
        dynamic_k=True,
        dynamic_coverage_threshold=args.confidence_threshold,
        dynamic_budget_ratios=ratios,
    )
    print(f"wrote {Path(args.out)}")


if __name__ == "__main__":
    main()
