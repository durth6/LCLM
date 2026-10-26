"""Materialize strict latent-only NIAH inputs from ``latent-context/lclm-eval``.

The original evaluation rows contain a complete LCLM prompt, with source text
between ``<|memory_start|>`` and ``<|memory_end|>``.  This utility splits each
row into two artifacts:

* ``memory.jsonl`` — offline cache-building input, containing ``context``;
* ``queries.jsonl`` — online input, containing a prompt with exactly one
  ``<|selected_memory|>`` marker and no source text.

The separation makes it impossible for ``inference.query_aware answer`` to
silently re-read the document.  ``answers`` are evaluation labels only; the
online selector does not receive them as model input.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any


RULER_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe", "qa_1", "qa_2",
]
NIAH_SUBTASKS = RULER_TASKS[:8]

MEMORY_START = "<|memory_start|>"
MEMORY_END = "<|memory_end|>"
SELECTED_MEMORY = "<|selected_memory|>"
MEMORY_PATTERN = re.compile(
    rf"{re.escape(MEMORY_START)}(.*?){re.escape(MEMORY_END)}", re.DOTALL
)
FWE_PREAMBLE = (
    "Read the following coded text and track the frequency of each coded word. "
    "Find the three most frequently appeared coded words. "
)
NIAH_TRAILING_QUERY_TASKS = {"niah_multivalue", "niah_multiquery"}
NIAH_TRAILING_QUERY_PREFIX = "\nWhat are all the special magic"


def _answers(row: dict[str, Any]) -> list[str]:
    ground_truth = (row.get("extra_info") or {}).get("ground_truth") or {}
    answers = ground_truth.get("answers") or row.get("answer") or row.get("answers") or []
    if isinstance(answers, str):
        answers = [answers]
    return [str(answer) for answer in answers]


def split_lclm_prompt(prompt: str, task: str | None = None) -> tuple[str, str, str]:
    """Return ``(context, selector_query, decoder_prompt)`` without leakage."""
    matches = list(MEMORY_PATTERN.finditer(prompt))
    if len(matches) == 1:
        match = matches[0]
        context = match.group(1)
        trailing_query = ""
        if task in NIAH_TRAILING_QUERY_TASKS:
            query_start = context.rfind(NIAH_TRAILING_QUERY_PREFIX)
            if query_start < 0:
                raise ValueError(f"could not identify the trailing {task} question inside memory")
            trailing_query = context[query_start:]
            context = context[:query_start]
            if "mentioned in the provided text" not in trailing_query:
                raise ValueError(f"invalid trailing {task} question")
        decoder_prompt = (
            prompt[:match.start()] + SELECTED_MEMORY + trailing_query + prompt[match.end():]
        )
    elif task == "fwe" and not matches:
        preamble_start = prompt.find(FWE_PREAMBLE)
        question_start = prompt.rfind("\n\nQuestion:")
        if preamble_start < 0 or question_start < 0:
            raise ValueError("could not identify the marker-free FWE context boundaries")
        context_start = preamble_start + len(FWE_PREAMBLE)
        if context_start >= question_start:
            raise ValueError("invalid marker-free FWE context range")
        context = prompt[context_start:question_start]
        decoder_prompt = prompt[:context_start] + SELECTED_MEMORY + prompt[question_start:]
    else:
        raise ValueError(f"expected exactly one memory region, found {len(matches)}")
    # The embedding model sees the short instruction/question only.  The
    # marker itself is removed to ensure no context text is carried online.
    selector_query = decoder_prompt.replace(SELECTED_MEMORY, "")
    if context in selector_query:
        raise AssertionError("source context leaked into selector query")
    return context, selector_query, decoder_prompt


def _teacher_target(answers: list[str], scoring_function: str) -> str:
    unique_answers = list(dict.fromkeys(answers))
    if not unique_answers:
        raise ValueError("RULER row has no answer")
    if scoring_function == "ruler_string_match_part":
        return unique_answers[0]
    if scoring_function != "ruler_string_match_all":
        raise ValueError(f"unsupported RULER scoring function: {scoring_function}")
    return ", ".join(unique_answers)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare leakage-safe RULER NIAH query-aware inputs")
    parser.add_argument("--ctx", type=int, default=4096, choices=[4096, 8192, 16384, 32768])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tasks", default="all",
                        help="comma-separated task names or 'all' for all 13 RULER tasks")
    parser.add_argument("--limit-per-task", type=int, default=0, help="0 keeps every example")
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = list(RULER_TASKS) if args.tasks.strip().lower() == "all" else [
        task.strip() for task in args.tasks.split(",") if task.strip()
    ]
    unknown = sorted(set(tasks) - set(RULER_TASKS))
    if unknown:
        raise ValueError(f"unknown NIAH task(s): {', '.join(unknown)}")

    dataset = load_dataset("latent-context/lclm-eval", "ruler", split="test")
    memory_records: list[dict[str, Any]] = []
    query_records: list[dict[str, Any]] = []
    task_counts: dict[str, int] = {}
    category_to_task = {f"memwrap/ruler/{task}_{args.ctx}": task for task in tasks}
    # One pass avoids scanning the full 39K-row eval dataset once per NIAH
    # subtask.  The per-task counts still preserve the original dataset order.
    selected_rows = dataset.filter(lambda row: row["category"] in category_to_task)
    task_counts = {task: 0 for task in tasks}
    for row in selected_rows:
        task = category_to_task[row["category"]]
        if args.limit_per_task and task_counts[task] >= args.limit_per_task:
            continue
        row_index = task_counts[task]
        task_counts[task] += 1
        context, selector_query, decoder_prompt = split_lclm_prompt(row["prompt"], task=task)
        example_id = f"{task}-{args.ctx}-{row_index:05d}"
        extra_info = row.get("extra_info") or {}
        scoring_function = extra_info.get("scoring_function", "ruler_string_match_all")
        answers = _answers(row)
        memory_records.append({
            "id": example_id,
            "document_id": example_id,
            "context": context,
            "task": task,
        })
        query_records.append({
            "id": example_id,
            "document_id": example_id,
            "task": task,
            "selector_query": selector_query,
            "prompt": decoder_prompt,
            "answers": answers,
            "teacher_target": _teacher_target(answers, scoring_function),
            "scoring_function": scoring_function,
        })

    _write_jsonl(output_dir / "memory.jsonl", memory_records)
    _write_jsonl(output_dir / "queries.jsonl", query_records)
    (output_dir / "manifest.json").write_text(json.dumps({
        "ctx": args.ctx,
        "tasks": task_counts,
        "num_contexts": len(memory_records),
        "num_queries": len(query_records),
        "limit_per_task": args.limit_per_task,
        "source_dataset": "latent-context/lclm-eval",
        "source_config": "ruler",
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "document_id_contract": "query document_id resolves exactly one offline cache record",
        "query_contract": "queries.jsonl contains no raw context; exactly one selected-memory marker",
    }, indent=2) + "\n")
    print(f"wrote {len(memory_records)} cache contexts and {len(query_records)} online queries to {output_dir}")


if __name__ == "__main__":
    main()
