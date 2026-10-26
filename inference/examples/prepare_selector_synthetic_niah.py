"""Build an independent, leakage-safe synthetic NIAH training split.

Filler text is fetched read-only from the WikiText-103 *training* split via
the configured Hugging Face mirror. Facts and questions are generated with a
separate seed, so this is not a copy of RULER's public test prompts.  The
output follows the same offline/online separation as the RULER preparation
tool: raw contexts are present only in ``memory.jsonl`` for one-time cache
building, while ``queries.jsonl`` has a short query and a selected-memory
marker only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random


DATASET = "Salesforce/wikitext"
CONFIG = "wikitext-103-raw-v1"
SPLIT = "train"
SELECTED_MEMORY = "<|selected_memory|>"


def _fetch_wikitext_documents(offset: int, pages: int) -> list[tuple[str, str]]:
    """Load a bounded WikiText slice through the configured HF mirror.

    WikiText exposes article titles as heading rows. Rows are coalesced into
    source documents before train/validation partitioning so no article can
    contribute filler text to both sides of the selector split.
    """
    from datasets import load_dataset

    stop = offset + pages * 100
    rows = load_dataset(DATASET, CONFIG, split=f"{SPLIT}[{offset}:{stop}]")
    documents: list[tuple[str, str]] = []
    pieces: list[str] = []
    document_start = offset
    for relative_index, row in enumerate(rows):
        text = str(row["text"]).strip()
        is_heading = text.startswith("=") and text.endswith("=")
        if is_heading and pieces:
            combined = "\n\n".join(pieces).strip()
            if len(combined) >= 128:
                documents.append((f"wikitext-train-{document_start}", combined))
            pieces = []
            document_start = offset + relative_index
        if text:
            pieces.append(text)
    if pieces:
        combined = "\n\n".join(pieces).strip()
        if len(combined) >= 128:
            documents.append((f"wikitext-train-{document_start}", combined))
    if len(documents) < 2:
        raise RuntimeError("WikiText slice produced fewer than two usable source documents")
    return documents


def _partition_source_documents(
    documents: list[tuple[str, str]], validation_fraction: float, split_seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    shuffled = list(documents)
    random.Random(split_seed).shuffle(shuffled)
    validation_count = max(1, int(len(shuffled) * validation_fraction))
    if validation_count >= len(shuffled):
        raise ValueError("validation split leaves no source documents for training")
    return shuffled[validation_count:], shuffled[:validation_count]


def _chat_prompt(question: str) -> str:
    return (
        "<|im_start|>system\n"
        "Answer the question precisely using the supplied latent memory.\n"
        "<|im_end|>\n<|im_start|>user\n"
        f"{SELECTED_MEMORY}\n\n{question}\n"
        "<|im_end|>\n<|im_start|>assistant\n"
    )


def _make_context(
    filler: list[str],
    rng: random.Random,
    target_chars: int,
    sample_index: int,
    *,
    pair_count: int | None = None,
    asked_count: int | None = None,
) -> tuple[str, str, list[str]]:
    """Create one single/multi-key fact-retrieval item without answer leakage."""
    pieces: list[str] = []
    cursor = (sample_index * 17 + rng.randrange(len(filler))) % len(filler)
    while sum(map(len, pieces)) < target_chars:
        pieces.append(filler[cursor % len(filler)])
        cursor += 1
    context = "\n\n".join(pieces)
    if pair_count is None:
        pair_count = rng.choices([1, 2, 3, 4], weights=[0.45, 0.30, 0.18, 0.07])[0]
    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    keys: list[str] = []
    while len(keys) < pair_count:
        key = f"archive-{rng.randrange(10_000_000):07d}"
        if key not in keys:
            keys.append(key)
    values: list[str] = []
    while len(values) < pair_count:
        value = f"{rng.randrange(1_000_000, 10_000_000):07d}"
        if value not in context and value not in values:
            values.append(value)
    # Regenerate a value if it happens to occur in the natural filler text:
    # oracle matching must identify only the injected fact.
    insertion_points = sorted(rng.sample(range(max(1, len(context))), pair_count), reverse=True)
    for insertion_point, key, value in zip(insertion_points, keys, values):
        fact = f"\n\n[ARCHIVE FACT] The verification code for {key} is {value}.\n\n"
        context = context[:insertion_point] + fact + context[insertion_point:]
    if asked_count is None:
        asked_keys = keys if pair_count <= 2 else keys[:rng.choice([2, 3])]
    else:
        if not 0 < asked_count <= pair_count:
            raise ValueError("asked_count must be in [1, pair_count]")
        asked_keys = rng.sample(keys, asked_count)
    asked_values = [values[keys.index(key)] for key in asked_keys]
    if len(asked_keys) == 1:
        question = f"What is the verification code for {asked_keys[0]}? Answer only with the code."
    else:
        question = (
            "What are the verification codes for " + ", ".join(asked_keys) +
            "? Answer only with the codes in the same order, separated by commas."
        )
    if any(answer in question for answer in asked_values):
        raise AssertionError("answer leaked into selector query")
    return context, question, asked_values


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an independent synthetic NIAH selector-training split")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-examples", type=int, default=256)
    parser.add_argument("--target-chars", type=int, default=16_000,
                        help="approximate 4K-token source context; exact tokens are recorded after caching")
    parser.add_argument("--source-offset", type=int, default=10_000)
    parser.add_argument("--source-pages", type=int, default=4)
    parser.add_argument("--seed", type=int, default=5201)
    parser.add_argument("--split-seed", type=int, default=6201)
    parser.add_argument("--validation-fraction", type=float, default=0.125)
    parser.add_argument("--id-prefix", default="synthetic-niah-train",
                        help="unique prefix when multiple length buckets are merged")
    args = parser.parse_args()
    if args.num_examples <= 1 or args.target_chars <= 0 or args.source_pages <= 0:
        raise ValueError("num examples, target chars, and source pages must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation fraction must be in (0, 1)")

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_documents = _fetch_wikitext_documents(args.source_offset, args.source_pages)
    train_documents, validation_documents = _partition_source_documents(
        source_documents, args.validation_fraction, args.split_seed
    )
    source_overlap = sorted(
        {record_id for record_id, _ in train_documents}
        & {record_id for record_id, _ in validation_documents}
    )
    if source_overlap:
        raise AssertionError(f"source-document leakage across splits: {source_overlap[:3]}")
    rng = random.Random(args.seed)
    memories, queries = [], []
    validation_examples = max(1, int(args.num_examples * args.validation_fraction))
    train_examples = args.num_examples - validation_examples
    for sample_index in range(args.num_examples):
        split = "train" if sample_index < train_examples else "validation"
        document_pool = train_documents if split == "train" else validation_documents
        filler = [text for _, text in document_pool]
        context, question, answers = _make_context(filler, rng, args.target_chars, sample_index)
        record_id = f"{args.id_prefix}-{sample_index:06d}"
        memories.append({
            "id": record_id,
            "document_id": record_id,
            "context": context,
            "task": "synthetic_niah_train",
            "split": split,
        })
        queries.append({
            "id": record_id,
            "document_id": record_id,
            "task": "synthetic_niah_train",
            "split": split,
            "selector_query": question,
            "prompt": _chat_prompt(question),
            "answers": answers,
            "teacher_target": ", ".join(answers),
            "scoring_function": "ruler_string_match_all",
        })
    _write_jsonl(output_dir / "memory.jsonl", memories)
    _write_jsonl(output_dir / "queries.jsonl", queries)
    (output_dir / "manifest.json").write_text(json.dumps({
        "split": "train",
        "purpose": "stage_a_selector_training_only; not RULER or LongBench evaluation",
        "source_dataset": DATASET,
        "source_config": CONFIG,
        "source_split": SPLIT,
        "source_offset": args.source_offset,
        "source_pages": args.source_pages,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
        "num_train_examples": train_examples,
        "num_validation_examples": validation_examples,
        "train_source_document_ids": [record_id for record_id, _ in train_documents],
        "validation_source_document_ids": [record_id for record_id, _ in validation_documents],
        "source_document_overlap": source_overlap,
        "num_examples": args.num_examples,
        "target_chars": args.target_chars,
        "online_contract": "queries.jsonl contains no raw source context",
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
    }, indent=2) + "\n")
    print(json.dumps({
        "out_dir": str(output_dir),
        "examples": len(memories),
        "source_documents": len(source_documents),
    }))


if __name__ == "__main__":
    main()
