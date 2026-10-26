"""Build offline evidence-role spans for the synthetic archive task.

The synthetic training generator inserts facts with a deliberately stable
format.  This utility resolves the key, answer, key-answer relation, and a
small supporting context window in *character* coordinates.  The spans are
kept in a sidecar rather than in ``queries.jsonl`` so the online answer path
still receives only a short question and a latent cache identifier.

The output is consumed by ``annotate_oracle_latents`` which maps the character
spans through the checkpoint tokenizer and then onto latent indices.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

from latent_context.records import document_id


ROLE_SCHEMA_VERSION = 1
_KEY_RE = re.compile(r"archive-\d{7}")
_FACT_RE = re.compile(
    r"\[ARCHIVE FACT\] The verification code for (?P<key>archive-\d{7}) "
    r"is (?P<answer>\d{7})\."
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _fact_index(context: str) -> dict[tuple[str, str], tuple[int, int, int, int]]:
    """Return ``(fact_start, fact_end, key_start, answer_start)`` by pair."""
    facts: dict[tuple[str, str], tuple[int, int, int, int]] = {}
    for match in _FACT_RE.finditer(context):
        key_start = match.start("key")
        answer_start = match.start("answer")
        pair = (match.group("key"), match.group("answer"))
        if pair in facts:
            raise ValueError(f"duplicate synthetic fact for {pair[0]} / {pair[1]}")
        facts[pair] = (match.start(), match.end(), key_start, answer_start)
    return facts


def _requested_keys(question: str) -> list[str]:
    keys = _KEY_RE.findall(question)
    if not keys:
        raise ValueError("synthetic selector question contains no archive key")
    # A key occurs once in a well-formed generated question.  Rejecting
    # duplicates avoids silently assigning one answer to two evidence groups.
    if len(set(keys)) != len(keys):
        raise ValueError("synthetic selector question repeats an archive key")
    return keys


def build_role_row(
    memory: dict[str, Any],
    query: dict[str, Any],
    *,
    support_radius: int = 96,
) -> dict[str, Any]:
    """Resolve one synthetic query into auditable character-span roles."""
    if support_radius < 0:
        raise ValueError("support_radius must be non-negative")
    context = memory.get("context")
    if not isinstance(context, str) or not context:
        raise ValueError(f"memory {memory.get('id')} has no source context")
    answers = [str(answer) for answer in query.get("answers", [])]
    keys = _requested_keys(str(query.get("selector_query", query.get("question", ""))))
    if len(keys) != len(answers):
        raise ValueError(
            f"query {query.get('id')} has {len(keys)} keys but {len(answers)} answers"
        )
    facts = _fact_index(context)
    groups: list[dict[str, Any]] = []
    for group_index, (key, answer) in enumerate(zip(keys, answers)):
        fact = facts.get((key, answer))
        if fact is None:
            raise ValueError(f"missing synthetic fact for {key} / {answer}")
        fact_start, fact_end, key_start, answer_start = fact
        answer_end = answer_start + len(answer)
        groups.append({
            "group": group_index,
            "key_char_spans": [[key_start, key_start + len(key)]],
            "answer_char_spans": [[answer_start, answer_end]],
            "key_answer_char_spans": [[fact_start, fact_end]],
            "support_char_spans": [[
                max(0, fact_start - support_radius),
                min(len(context), fact_end + support_radius),
            ]],
        })

    def flatten(role: str) -> list[list[int]]:
        return [span for group in groups for span in group[f"{role}_char_spans"]]

    return {
        "id": str(query["id"]),
        "document_id": document_id(query),
        "role_span_schema_version": ROLE_SCHEMA_VERSION,
        "source": "synthetic_archive_fact_v1",
        "support_radius": support_radius,
        "key_char_spans": flatten("key"),
        "answer_char_spans": flatten("answer"),
        "key_answer_char_spans": flatten("key_answer"),
        "support_char_spans": flatten("support"),
        "role_groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate synthetic archive evidence roles offline")
    parser.add_argument("--memory-jsonl", required=True)
    parser.add_argument("--queries-jsonl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--support-radius", type=int, default=96)
    args = parser.parse_args()
    memories = {}
    for row in _read_jsonl(args.memory_jsonl):
        memory_id = document_id(row)
        if memory_id in memories:
            raise ValueError(f"duplicate memory document ID: {memory_id}")
        memories[memory_id] = row
    rows = []
    seen: set[str] = set()
    for query in _read_jsonl(args.queries_jsonl):
        query_id = str(query["id"])
        if query_id in seen:
            raise ValueError(f"duplicate query ID: {query_id}")
        seen.add(query_id)
        memory_id = document_id(query)
        if memory_id not in memories:
            raise ValueError(f"query {query_id} references missing document {memory_id}")
        rows.append(build_role_row(memories[memory_id], query, support_radius=args.support_radius))
    _write_jsonl(args.out, rows)
    print(json.dumps({"records": len(rows), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
