"""Build an independent hard multi-key validation set for dynamic-K calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from inference.examples.prepare_selector_synthetic_niah import (
    CONFIG,
    DATASET,
    SPLIT,
    _chat_prompt,
    _fetch_wikitext_documents,
    _make_context,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--examples-per-length", type=int, default=64)
    parser.add_argument("--length-tags", default="4k,8k,16k,32k")
    parser.add_argument("--target-chars", default="16000,32000,64000,128000")
    parser.add_argument("--source-offsets", default="100000,110000,120000,130000")
    parser.add_argument("--source-pages", type=int, default=4)
    parser.add_argument("--pair-count", type=int, default=24)
    parser.add_argument("--asked-counts", default="2,4,8,12,16")
    parser.add_argument("--seed", type=int, default=8301)
    args = parser.parse_args()
    tags = [value for value in args.length_tags.split(",") if value]
    lengths = [int(value) for value in args.target_chars.split(",") if value]
    offsets = [int(value) for value in args.source_offsets.split(",") if value]
    asked_counts = [int(value) for value in args.asked_counts.split(",") if value]
    if not (len(tags) == len(lengths) == len(offsets)):
        raise ValueError("length tags, target chars and source offsets must align")
    if args.examples_per_length <= 0 or args.source_pages <= 0 or args.pair_count <= 0:
        raise ValueError("example, source-page and pair counts must be positive")
    if not asked_counts or min(asked_counts) <= 0 or max(asked_counts) > args.pair_count:
        raise ValueError("asked counts must be nonempty and no larger than pair count")

    memories: list[dict] = []
    queries: list[dict] = []
    components = []
    for length_index, (tag, target_chars, offset) in enumerate(zip(tags, lengths, offsets)):
        source_documents = _fetch_wikitext_documents(offset, args.source_pages)
        filler = [text for _, text in source_documents]
        rng = random.Random(args.seed + length_index)
        counts = []
        for sample_index in range(args.examples_per_length):
            asked_count = asked_counts[sample_index % len(asked_counts)]
            context, question, answers = _make_context(
                filler,
                rng,
                target_chars,
                sample_index,
                pair_count=args.pair_count,
                asked_count=asked_count,
            )
            record_id = f"dynamic-k-cal-{tag}-{sample_index:06d}"
            memories.append({
                "id": record_id,
                "document_id": record_id,
                "context": context,
                "task": "synthetic_dynamic_k_validation",
                "split": "validation",
                "length_tag": tag,
            })
            queries.append({
                "id": record_id,
                "document_id": record_id,
                "task": "synthetic_dynamic_k_validation",
                "split": "validation",
                "length_tag": tag,
                "selector_query": question,
                "prompt": _chat_prompt(question),
                "answers": answers,
                "teacher_target": ", ".join(answers),
                "scoring_function": "ruler_string_match_all",
            })
            counts.append(asked_count)
        components.append({
            "length_tag": tag,
            "target_chars": target_chars,
            "source_offset": offset,
            "source_pages": args.source_pages,
            "source_document_ids": [record_id for record_id, _ in source_documents],
            "asked_count_distribution": {
                str(value): counts.count(value) for value in sorted(set(counts))
            },
        })

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "memory.jsonl", memories)
    _write_jsonl(output / "queries.jsonl", queries)
    (output / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "split": "validation",
        "purpose": "dynamic_k_calibration_only; never RULER test evaluation",
        "source_dataset": DATASET,
        "source_config": CONFIG,
        "source_split": SPLIT,
        "seed": args.seed,
        "pair_count": args.pair_count,
        "asked_counts": asked_counts,
        "num_examples": len(memories),
        "components": components,
        "online_contract": "queries.jsonl contains no raw source context",
    }, indent=2) + "\n")
    print(json.dumps({"out_dir": str(output), "examples": len(memories)}))


if __name__ == "__main__":
    main()
