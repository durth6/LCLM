"""Evaluate a role-aware selector through the standard latent-only decoder path."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import inference.query_aware as query_aware
from latent_context.role_selector import load_role_selector_checkpoint
from inference.examples.eval_query_aware_niah import (
    _attach_role_metrics,
    _build_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--selector-checkpoint", required=True)
    parser.add_argument("--role-jsonl", required=True)
    parser.add_argument("--selection-ratio", type=float, default=0.25)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--random-seed", type=int, default=6201)
    parser.add_argument("--position-policy", default="full_layout", choices=("compact", "full_layout"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--reuse-queries", type=int, default=1)
    parser.add_argument("--repeat-each-query", type=int, default=1)
    parser.add_argument("--max-resident-caches", type=int, default=1)
    parser.add_argument("--compute-answer-nll", action="store_true")
    parser.add_argument("--dynamic-k", action="store_true")
    parser.add_argument("--dynamic-threshold", type=float, default=0.90)
    args = parser.parse_args()

    # Load once in this process and make the standard answer function use the
    # role-aware checkpoint. This preserves all existing cache/fingerprint,
    # position and latency accounting without introducing a text fallback.
    model, _, _ = query_aware.from_pretrained(args.checkpoint, device=args.device, dtype=args.dtype)
    selector, metadata = load_role_selector_checkpoint(
        args.selector_checkpoint, model.memory_fingerprint()
    )
    selector = selector.to(args.device)
    selector.eval()
    query_aware.load_selector_checkpoint = lambda _path, _fingerprint: (selector, None, metadata)

    ratios = (0.25, 0.5, 0.75, 1.0)
    query_aware.answer_queries(
        checkpoint=args.checkpoint,
        cache_dir=args.cache_dir,
        queries_jsonl=str(Path(args.input_dir) / "queries.jsonl"),
        out=str(Path(args.work_dir) / "answers.jsonl"),
        k_decoder=None,
        selection_ratio=args.selection_ratio,
        max_tokens=args.max_tokens,
        device=args.device,
        dtype=args.dtype,
        selector_mode="learned",
        selector_checkpoint=args.selector_checkpoint,
        oracle_jsonl=args.role_jsonl,
        random_seed=args.random_seed,
        max_resident_caches=args.max_resident_caches,
        repeat_each_query=args.repeat_each_query,
        position_policy=args.position_policy,
        compute_answer_nll=args.compute_answer_nll,
        dynamic_k=args.dynamic_k,
        dynamic_coverage_threshold=args.dynamic_threshold,
        dynamic_budget_ratios=ratios,
    )

    work = Path(args.work_dir)
    records = [json.loads(line) for line in (work / "answers.jsonl").read_text().splitlines() if line.strip()]
    records = _attach_role_metrics(records, args.role_jsonl)
    (work / "answers.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    )
    manifest = json.loads((Path(args.cache_dir) / "manifest.json").read_text())
    ids = {str(row.get("document_id", row["id"])) for row in records}
    manifest = [row for row in manifest if str(row.get("document_id", row["id"])) in ids]
    summary_args = Namespace(
        checkpoint=args.checkpoint,
        k_decoder=None,
        selection_ratio=args.selection_ratio,
        selector_mode="role_aware",
        neighbor_radius=0,
        repeat_each_query=args.repeat_each_query,
        max_resident_caches=args.max_resident_caches,
        reuse_queries=args.reuse_queries,
        position_policy=args.position_policy,
        oracle_role=None,
        role_jsonl=args.role_jsonl,
        coverage_rerank=False,
        coverage_weight=0.0,
        dynamic_k=args.dynamic_k,
        dynamic_coverage_threshold=args.dynamic_threshold,
    )
    summary = _build_summary(summary_args, manifest, records)
    summary["role_selector_checkpoint"] = args.selector_checkpoint
    summary["role_selector_metadata"] = metadata
    (work / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"work_dir": str(work), "queries": len(records), "ruler_accuracy": summary["ruler_accuracy"]}))


if __name__ == "__main__":
    main()
