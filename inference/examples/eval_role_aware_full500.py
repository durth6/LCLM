"""Evaluate RoleAwareSelector using the established latent-only answer path."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import inference.query_aware as query_aware
from inference.examples.eval_query_aware_niah import _attach_role_metrics, _build_summary
from latent_context.role_selector import load_role_selector_checkpoint


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
    parser.add_argument("--compute-answer-nll", action="store_true")
    args = parser.parse_args()

    role_checkpoint = args.selector_checkpoint
    original_loader = query_aware.load_selector_checkpoint

    def load_role_selector(_path: str, expected_fingerprint: str):
        selector, metadata = load_role_selector_checkpoint(role_checkpoint, expected_fingerprint)
        return selector, None, metadata

    # answer_queries itself loads the LCLM exactly once. The injected loader
    # receives that model's fingerprint, so compatibility remains fail-closed.
    query_aware.load_selector_checkpoint = load_role_selector
    try:
        work = Path(args.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        query_aware.answer_queries(
            checkpoint=args.checkpoint,
            cache_dir=args.cache_dir,
            queries_jsonl=str(Path(args.input_dir) / "queries.jsonl"),
            out=str(work / "answers.jsonl"),
            k_decoder=None,
            selection_ratio=args.selection_ratio,
            max_tokens=args.max_tokens,
            device=args.device,
            dtype=args.dtype,
            selector_mode="learned",
            selector_checkpoint=role_checkpoint,
            oracle_jsonl=args.role_jsonl,
            random_seed=args.random_seed,
            max_resident_caches=1,
            repeat_each_query=1,
            position_policy=args.position_policy,
            compute_answer_nll=args.compute_answer_nll,
        )
    finally:
        query_aware.load_selector_checkpoint = original_loader

    records = [
        json.loads(line) for line in (work / "answers.jsonl").read_text().splitlines()
        if line.strip()
    ]
    records = _attach_role_metrics(records, args.role_jsonl)
    (work / "answers.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    )
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
        dynamic_k=False,
        dynamic_coverage_threshold=None,
        repeat_each_query=1,
        max_resident_caches=1,
        reuse_queries=1,
    )
    summary = _build_summary(summary_args, manifest, records)
    summary["role_selector_checkpoint"] = role_checkpoint
    (work / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "work_dir": str(work),
        "queries": len(records),
        "ruler_accuracy": summary["ruler_accuracy"],
    }))


if __name__ == "__main__":
    main()
