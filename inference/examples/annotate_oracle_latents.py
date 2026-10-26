"""Create offline answer-span oracle labels for cached RULER latent memories.

The output is a sidecar evaluation/training artifact, never part of a
``LatentMemory`` cache.  Online selectors therefore cannot access oracle
labels or source context at inference time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from latent_context.records import document_id
from latent_context.selector import load_latent_memory


ORACLE_SCHEMA_VERSION = 1


def find_answer_token_spans(
    context: str,
    answers: list[str],
    offsets: list[tuple[int, int]],
) -> tuple[list[list[int]], int]:
    """Map all exact answer-string occurrences to nonempty token spans."""
    per_answer, matched_answers = find_answer_token_spans_by_answer(context, answers, offsets)
    return [span for answer_spans in per_answer for span in answer_spans], matched_answers


def find_answer_token_spans_by_answer(
    context: str,
    answers: list[str],
    offsets: list[tuple[int, int]],
) -> tuple[list[list[list[int]]], int]:
    """Return token spans grouped by answer, preserving all occurrences."""
    per_answer: list[list[list[int]]] = []
    matched_answers = 0
    for answer in answers:
        answer_spans: list[list[int]] = []
        start = 0
        while True:
            char_start = context.find(answer, start)
            if char_start < 0:
                break
            char_end = char_start + len(answer)
            token_indices = [
                token_index for token_index, (offset_start, offset_end) in enumerate(offsets)
                if offset_start < char_end and offset_end > char_start
            ]
            if token_indices:
                answer_spans.append([token_indices[0], token_indices[-1] + 1])
            start = char_start + max(len(answer), 1)
        per_answer.append(answer_spans)
        matched_answers += int(bool(answer_spans))
    return per_answer, matched_answers


def map_token_spans_to_latents(token_spans: list[list[int]], memory) -> list[int]:
    selected = set()
    starts = memory.latent_start.tolist()
    ends = memory.latent_end.tolist()
    for token_start, token_end in token_spans:
        for latent_index, (latent_start, latent_end) in enumerate(zip(starts, ends)):
            if latent_start < token_end and latent_end > token_start:
                selected.add(latent_index)
    return sorted(selected)


def _normalize_spans(raw_spans: Any) -> list[list[int]]:
    """Normalize a JSON span list without accepting malformed coordinates."""
    if not isinstance(raw_spans, list):
        return []
    normalized: list[list[int]] = []
    for span in raw_spans:
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        start, end = int(span[0]), int(span[1])
        if start < 0 or end <= start:
            continue
        normalized.append([start, end])
    return normalized


def _char_spans_to_token_spans(
    char_spans: list[list[int]], offsets: list[tuple[int, int]],
) -> list[list[int]]:
    token_spans: list[list[int]] = []
    for char_start, char_end in char_spans:
        token_indices = [
            token_index
            for token_index, (offset_start, offset_end) in enumerate(offsets)
            if offset_start < char_end and offset_end > char_start
        ]
        if token_indices:
            token_spans.append([token_indices[0], token_indices[-1] + 1])
    return token_spans


def _optional_role_spans(
    query: dict[str, Any],
    memory,
    offsets: list[tuple[int, int]] | None = None,
    role_row: dict[str, Any] | None = None,
) -> dict[str, list[int]]:
    """Map optional offline key/support spans into latent-role labels.

    Standard RULER rows expose answer strings only; these roles are populated
    when an offline synthetic or task-specific role sidecar supplies richer
    spans.  Keeping them in a sidecar avoids changing the online query
    contract.
    """
    roles: dict[str, list[int]] = {}
    role_row = role_row or {}
    if offsets is None:
        offsets = []

    def add_role(role: str, token_spans: Any = None, char_spans: Any = None) -> None:
        normalized_tokens = _normalize_spans(token_spans)
        if not normalized_tokens and offsets:
            normalized_tokens = _char_spans_to_token_spans(
                _normalize_spans(char_spans), offsets
            )
        if normalized_tokens:
            roles[role] = map_token_spans_to_latents(normalized_tokens, memory)

    role_names = {"key", "support", "key_answer", "all_answer"}
    for source in (query, role_row):
        for key, value in source.items():
            if key.endswith("_token_spans"):
                role_names.add(key.removesuffix("_token_spans"))
            elif key.endswith("_char_spans"):
                role_names.add(key.removesuffix("_char_spans"))
        role_mapping = source.get("role_token_spans")
        if isinstance(role_mapping, dict):
            role_names.update(str(role) for role in role_mapping)
        role_mapping = source.get("role_char_spans")
        if isinstance(role_mapping, dict):
            role_names.update(str(role) for role in role_mapping)

    for role in sorted(role_names):
        token_value = None
        char_value = None
        for source in (query, role_row):
            if token_value is None:
                token_value = source.get(f"{role}_token_spans")
            if char_value is None:
                char_value = source.get(f"{role}_char_spans")
            token_mapping = source.get("role_token_spans")
            char_mapping = source.get("role_char_spans")
            if token_value is None and isinstance(token_mapping, dict):
                token_value = token_mapping.get(role)
            if char_value is None and isinstance(char_mapping, dict):
                char_value = char_mapping.get(role)
        add_role(role, token_value, char_value)

    # ``role_groups`` is useful for multi-key supervision: each group gets a
    # distinct mask while the base role keeps the union for diagnostics.
    for group_index, group in enumerate(role_row.get("role_groups", [])):
        if not isinstance(group, dict):
            continue
        for role in ("key", "answer", "key_answer", "support", "all_answer"):
            add_role(
                f"{role}_{group_index}",
                group.get(f"{role}_token_spans"),
                group.get(f"{role}_char_spans"),
            )
    return roles


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate answer-containing LCLM latents offline")
    parser.add_argument("--checkpoint", required=True, help="local LCLM checkpoint; only encoder tokenizer is used")
    parser.add_argument("--memory-jsonl", required=True)
    parser.add_argument("--queries-jsonl", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--role-spans-jsonl",
        help="optional offline role sidecar with token- or character-coordinate spans",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(Path(args.checkpoint) / "encoder")
    contexts: dict[str, str] = {}
    for record in _read_jsonl(Path(args.memory_jsonl)):
        memory_id = document_id(record)
        if memory_id in contexts:
            raise ValueError(f"duplicate memory document ID: {memory_id}")
        contexts[memory_id] = record["context"]
    queries = _read_jsonl(Path(args.queries_jsonl))
    role_rows: dict[str, dict[str, Any]] = {}
    if args.role_spans_jsonl:
        for row in _read_jsonl(Path(args.role_spans_jsonl)):
            role_id = str(row["id"])
            if role_id in role_rows:
                raise ValueError(f"duplicate role sidecar row for {role_id}")
            role_rows[role_id] = row
    cache_dir = Path(args.cache_dir)
    output_rows = []
    covered_answers = 0
    total_answers = 0
    seen_query_ids: set[str] = set()
    for query in queries:
        record_id = str(query["id"])
        if record_id in seen_query_ids:
            raise ValueError(f"duplicate query ID: {record_id}")
        seen_query_ids.add(record_id)
        memory_id = document_id(query)
        if memory_id not in contexts:
            raise ValueError(f"query {record_id} references missing document {memory_id}")
        context = contexts[memory_id]
        answers = [str(answer) for answer in query.get("answers", [])]
        encoded = tokenizer(context, add_special_tokens=False, return_offsets_mapping=True)
        offsets = [tuple(offset) for offset in encoded["offset_mapping"]]
        memory = load_latent_memory(cache_dir / memory_id)
        token_count = len(encoded["input_ids"])
        if token_count != int(memory.latent_end.max()):
            raise ValueError(
                f"tokenization/cache mismatch for {record_id}: {token_count} != {int(memory.latent_end.max())}"
            )
        per_answer_spans, matched_answers = find_answer_token_spans_by_answer(
            context, answers, offsets
        )
        token_spans = [span for answer_spans in per_answer_spans for span in answer_spans]
        oracle_latents = map_token_spans_to_latents(token_spans, memory)
        primary_answer_latents = map_token_spans_to_latents(
            per_answer_spans[0] if per_answer_spans else [], memory
        )
        role_row = role_rows.get(record_id)
        if role_row is not None and role_row.get("document_id") is not None:
            if str(role_row["document_id"]) != memory_id:
                raise ValueError(f"role sidecar {record_id} points to a different document")
        role_latents = _optional_role_spans(query, memory, offsets, role_row)
        role_latents.setdefault("answer", primary_answer_latents)
        role_latents.setdefault("all_answer", oracle_latents)
        if "key" in role_latents:
            role_latents.setdefault(
                "key_answer",
                sorted(set(role_latents["key"]) | set(oracle_latents)),
            )
        output_rows.append({
            "id": record_id,
            "document_id": memory_id,
            "oracle_schema_version": ORACLE_SCHEMA_VERSION,
            "task": query.get("task"),
            "answer_count": len(answers),
            "matched_answer_count": matched_answers,
            "answer_token_spans": token_spans,
            "oracle_latent_indices": oracle_latents,
            "answer_latent_indices": primary_answer_latents,
            "all_answer_latent_indices": role_latents.get("all_answer", oracle_latents),
            "role_latent_indices": role_latents,
            "cache_fingerprint": memory.fingerprint,
            "cache_schema_version": memory.schema_version,
            "role_span_schema_version": (
                role_row.get("role_span_schema_version") if role_row else None
            ),
            "role_labels_available": bool(role_row and role_row.get("roles_available", True)),
        })
        covered_answers += matched_answers
        total_answers += len(answers)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "records": len(output_rows),
        "answer_match_recall": covered_answers / total_answers if total_answers else 0.0,
        "records_with_oracle_latents": sum(bool(row["oracle_latent_indices"]) for row in output_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
