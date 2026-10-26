"""Build offline evidence-role spans for the prepared RULER inputs.

The normal RULER oracle sidecar only records answer-containing latents.  That
is enough for a localization diagnostic, but it cannot tell whether a
selector retained the key and the local relation that makes a multi-evidence
question answerable.  This helper extracts those spans from the *offline*
source memory and writes a separate character-span sidecar.  The sidecar is
then mapped to latent indices by :mod:`annotate_oracle_latents`.

No row produced here is consumed by the online answer path.  In particular,
the prepared ``queries.jsonl`` remains free of source text and oracle labels.
The parser is deliberately limited to the structured NIAH fact format; for
the other RULER task families it emits an explicit unavailable row instead of
guessing evidence boundaries.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

from latent_context.records import document_id


ROLE_SPAN_SCHEMA_VERSION = 1
NIAH_TASKS = {
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
}

# The answer values in the NIAH generators do not contain a period.  Keeping
# the sentence boundary in the regex gives us a useful, auditable support
# group without trying to infer document structure from arbitrary prose.
_FACT_RE = re.compile(
    r"One of the special magic (?P<kind>numbers|uuids) for "
    r"(?P<key>[^\.\n]+?) is: (?P<answer>[^\.\n]+)\."
)
_QUESTION_KEYS_RE = re.compile(
    r"(?:What|Which)\s+(?:is|are)\s+(?:all\s+)?the\s+special\s+magic\s+"
    r"(?:number|numbers|uuid|uuids)\s+for\s+(?P<keys>.+?)\s+mentioned\s+"
    r"in\s+the\s+provided\s+text",
    re.IGNORECASE | re.DOTALL,
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _query_keys(query_text: str) -> list[str]:
    """Return NIAH keys from the question clause, not generic prompt prose.

    The selector query includes a system instruction containing words such as
    ``long-context``.  Searching the whole prompt for hyphenated tokens would
    therefore create false keys before reaching the actual question.  RULER's
    NIAH templates have a stable ``magic ... for ... mentioned`` clause, so we
    restrict extraction to that clause and retain its source order.
    """
    question_match = _QUESTION_KEYS_RE.search(query_text)
    if question_match is None:
        raise ValueError("could not locate the structured NIAH question clause")
    key_text = question_match.group("keys").strip()
    # Multi-query templates separate keys with commas and put ``and`` before
    # the final item.  A key itself may contain spaces, so token-level regex
    # extraction would incorrectly split e.g. ``capricious-creme brulee``.
    parts = [part.strip() for part in key_text.split(",")]
    if len(parts) == 1:
        # A few RULER renderings omit the comma before the final ``and``.
        # Generated keys are adjective-noun phrases, so splitting this
        # conjunction is unambiguous while preserving spaces within a key
        # such as ``capricious-creme brulee``.
        parts = [
            part.strip()
            for part in re.split(r"\s+(?:and|or)\s+", parts[0], flags=re.IGNORECASE)
        ]
    else:
        parts[-1] = re.sub(r"^(?:and|or)\s+", "", parts[-1], flags=re.IGNORECASE)
    keys: list[str] = []
    seen: set[str] = set()
    for key in parts:
        key = key.strip()
        if not key:
            continue
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def _fact_rows(context: str) -> list[re.Match[str]]:
    return list(_FACT_RE.finditer(context))


def build_role_row(
    memory: dict[str, Any],
    query: dict[str, Any],
    *,
    support_radius: int = 96,
) -> dict[str, Any]:
    """Resolve one prepared RULER query into character-coordinate roles.

    NIAH rows get one group per requested key.  A multi-value key can have
    several fact sentences, so all matching sentences are retained in that
    group.  Unsupported task families are represented explicitly with
    ``roles_available=False``.
    """
    if support_radius < 0:
        raise ValueError("support_radius must be non-negative")
    query_id = str(query["id"])
    memory_id = document_id(query)
    if document_id(memory) != memory_id:
        raise ValueError(f"query {query_id} points to a different document")
    context = memory.get("context")
    if not isinstance(context, str) or not context:
        raise ValueError(f"memory {memory_id} has no source context")

    base = {
        "id": query_id,
        "document_id": memory_id,
        "task": query.get("task"),
        "role_span_schema_version": ROLE_SPAN_SCHEMA_VERSION,
        "support_radius": support_radius,
        "source": "ruler_niah_fact_v1",
        "roles_available": False,
        "role_char_spans": {},
        "role_groups": [],
    }
    if query.get("task") not in NIAH_TASKS:
        return base

    query_text = str(query.get("selector_query", query.get("question", "")))
    keys = _query_keys(query_text)
    if not keys:
        raise ValueError(f"NIAH query {query_id} contains no hyphenated key")
    answers = [str(answer) for answer in query.get("answers", [])]
    facts = _fact_rows(context)
    groups: list[dict[str, Any]] = []
    for group_index, key in enumerate(keys):
        key_facts = [fact for fact in facts if fact.group("key") == key]
        if answers:
            # Match by answer as well as key.  This prevents a repeated key in
            # a malformed row from silently assigning a distractor sentence.
            matching = [fact for fact in key_facts if fact.group("answer") in answers]
            if matching:
                key_facts = matching
        if not key_facts:
            raise ValueError(f"could not resolve NIAH key {key!r} for {query_id}")

        key_spans: list[list[int]] = []
        answer_spans: list[list[int]] = []
        key_answer_spans: list[list[int]] = []
        support_spans: list[list[int]] = []
        for fact in key_facts:
            key_spans.append(list(fact.span("key")))
            answer_spans.append(list(fact.span("answer")))
            key_answer_spans.append(list(fact.span()))
            support_spans.append([
                max(0, fact.start() - support_radius),
                min(len(context), fact.end() + support_radius),
            ])
        groups.append({
            "group": group_index,
            "key": key,
            "key_char_spans": key_spans,
            "answer_char_spans": answer_spans,
            "key_answer_char_spans": key_answer_spans,
            "support_char_spans": support_spans,
        })

    def flatten(name: str) -> list[list[int]]:
        return [span for group in groups for span in group[name]]

    all_answers = flatten("answer_char_spans")
    base.update({
        "roles_available": True,
        "role_char_spans": {
            "key": flatten("key_char_spans"),
            # Preserve the historical primary-answer convention while making
            # the complete answer union available explicitly.
            "answer": groups[0]["answer_char_spans"],
            "all_answer": all_answers,
            "key_answer": flatten("key_answer_char_spans"),
            "support": flatten("support_char_spans"),
        },
        "role_groups": groups,
    })
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate offline RULER key/answer/support spans")
    parser.add_argument("--memory-jsonl", required=True)
    parser.add_argument("--queries-jsonl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--support-radius", type=int, default=96)
    args = parser.parse_args()

    memories: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(args.memory_jsonl):
        memory_id = document_id(row)
        if memory_id in memories:
            raise ValueError(f"duplicate memory document ID: {memory_id}")
        memories[memory_id] = row

    rows: list[dict[str, Any]] = []
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
    print(json.dumps({
        "records": len(rows),
        "roles_available": sum(bool(row["roles_available"]) for row in rows),
        "out": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
