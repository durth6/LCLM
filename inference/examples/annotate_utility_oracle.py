"""Build an offline answer-NLL utility oracle over an explicit candidate family.

The global optimum over all ``M choose K`` latent subsets is intractable at the
target lengths. This tool therefore defines a finite, auditable candidate
family and returns its exact minimum teacher-forced answer NLL. The output
explicitly records that scope and must not be described as a global oracle.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from inference.examples.annotate_contribution_latents import teacher_forced_nll
from latent_context import (
    DEFAULT_POSITION_POLICY,
    POSITION_POLICIES,
    answer_nll_metrics,
    document_id,
    exact_group_oracle_candidates,
    from_pretrained,
    select_uniform_indices,
)
from latent_context.selector import load_latent_memory


UTILITY_ORACLE_SCHEMA_VERSION = 2
UTILITY_ORACLE_SCOPE = "exact_answer_nll_minimum_over_declared_candidates_not_global_subset_optimum"


def _candidate_roles(names: list[str]) -> list[str]:
    """Map candidate definitions to auditable evidence-role labels."""
    roles: set[str] = set()
    for name in names:
        lowered = str(name).lower()
        if "span" in lowered or "answer" in lowered:
            roles.add("answer")
        if "key" in lowered:
            roles.add("key")
        if "group" in lowered or "neighbor" in lowered or "uniform" in lowered:
            roles.add("support")
        if "learned" in lowered or "contribution" in lowered:
            roles.add("utility")
    return sorted(roles or {"candidate"})


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _rows_by_id(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        record_id = str(row["id"])
        if record_id in rows:
            raise ValueError(f"duplicate sidecar row for {record_id}")
        rows[record_id] = row
    return rows


def _parse_named_sidecars(values: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    sidecars = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--candidate-jsonl values must use NAME=PATH")
        name, raw_path = value.split("=", 1)
        if not name or name in sidecars:
            raise ValueError(f"invalid or duplicate candidate name: {name!r}")
        sidecars[name] = _rows_by_id(raw_path)
    return sidecars


def _span_fill_indices(positives: Iterable[int], memory_length: int, k: int) -> tuple[int, ...]:
    chosen = {int(index) for index in positives if 0 <= int(index) < memory_length}
    if len(chosen) >= k:
        return tuple(sorted(chosen)[:k])
    for index in select_uniform_indices(memory_length, min(memory_length, max(k * 4, k))).tolist():
        chosen.add(int(index))
        if len(chosen) == k:
            break
    for index in range(memory_length):
        if len(chosen) == k:
            break
        chosen.add(index)
    return tuple(sorted(chosen))


def _span_neighbor_indices(
    positives: Iterable[int], memory_length: int, k: int, radius: int = 1,
) -> tuple[int, ...]:
    """Expand answer-span seeds with local source neighbors under exact K."""
    if radius < 0:
        raise ValueError("neighbor radius must be non-negative")
    chosen = {
        index
        for raw_index in positives
        if 0 <= (index := int(raw_index)) < memory_length
    }
    seeds = sorted(chosen)
    for seed in seeds:
        for index in range(max(0, seed - radius), min(memory_length, seed + radius + 1)):
            if len(chosen) >= k:
                break
            chosen.add(index)
        if len(chosen) >= k:
            break
    if len(chosen) < k:
        for index in select_uniform_indices(memory_length, min(memory_length, max(k * 4, k))).tolist():
            chosen.add(int(index))
            if len(chosen) >= k:
                break
    if len(chosen) < k:
        chosen.update(range(memory_length))
    return tuple(sorted(chosen)[:k])


def build_candidate_family(
    memory_length: int,
    k: int,
    selection_ratio: float,
    num_groups: int,
    contribution_values: list[float] | None = None,
    span_indices: list[int] | None = None,
    external_candidates: dict[str, list[int]] | None = None,
    neighbor_radius: int = 1,
    role_indices: Mapping[str, Iterable[int]] | None = None,
) -> dict[tuple[int, ...], list[str]]:
    """Return unique exact-K subsets mapped to all definitions producing them."""
    candidates: dict[tuple[int, ...], list[str]] = {}

    def add(name: str, raw_indices: Iterable[int]) -> None:
        indices = tuple(int(index) for index in raw_indices)
        if len(indices) != k or tuple(sorted(set(indices))) != indices:
            raise ValueError(f"candidate {name!r} must contain exactly K unique source-ordered indices")
        if indices and (indices[0] < 0 or indices[-1] >= memory_length):
            raise IndexError(f"candidate {name!r} contains an out-of-range latent index")
        candidates.setdefault(indices, []).append(name)

    for candidate_index, indices in enumerate(
        exact_group_oracle_candidates(memory_length, selection_ratio, num_groups)
    ):
        add(f"group{num_groups}_combination_{candidate_index:03d}", indices)
    add("uniform", select_uniform_indices(memory_length, k).tolist())

    if contribution_values is not None:
        if len(contribution_values) != memory_length:
            raise ValueError("contribution target length does not match memory")
        values = torch.tensor(contribution_values, dtype=torch.float32)
        if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
            raise ValueError("contribution values must be finite and non-negative")
        add("contribution_topk", torch.sort(torch.argsort(values, descending=True, stable=True)[:k]).values.tolist())
    if span_indices is not None:
        add("span_oracle_fill", _span_fill_indices(span_indices, memory_length, k))
        add(
            f"span_oracle_neighbor_r{neighbor_radius}",
            _span_neighbor_indices(span_indices, memory_length, k, neighbor_radius),
        )
    if role_indices:
        normalized_roles = {
            str(role): sorted({int(index) for index in indices if 0 <= int(index) < memory_length})
            for role, indices in role_indices.items()
        }
        normalized_roles = {role: indices for role, indices in normalized_roles.items() if indices}
        for role in ("answer", "all_answer", "key", "support", "key_answer"):
            if role in normalized_roles:
                add(
                    f"{role}_role_fill",
                    _span_fill_indices(normalized_roles[role], memory_length, k),
                )
        role_unions = (
            ("key_answer_support_union", ("key", "answer", "support")),
            ("key_answer_union", ("key", "answer")),
            ("answer_support_union", ("answer", "support")),
        )
        for name, roles in role_unions:
            union = sorted({
                index for role in roles for index in normalized_roles.get(role, [])
            })
            if union:
                add(name, _span_fill_indices(union, memory_length, k))
    for name, indices in (external_candidates or {}).items():
        add(name, indices)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate an exact candidate-family answer-NLL oracle")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--queries-jsonl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--selection-ratio", type=float, default=0.25)
    parser.add_argument("--num-groups", type=int, default=8)
    parser.add_argument("--span-neighbor-radius", type=int, default=1)
    parser.add_argument("--contribution-jsonl")
    parser.add_argument("--span-oracle-jsonl")
    parser.add_argument(
        "--candidate-jsonl",
        action="append",
        default=[],
        help="additional exact-K selections as NAME=PATH; rows need selected_latent_indices",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-per-task", type=int, default=0,
                        help="0 keeps all; otherwise keeps at most N queries from each task")
    parser.add_argument("--max-target-tokens", type=int, default=0)
    parser.add_argument("--answer-separator", default=", ")
    parser.add_argument("--position-policy", choices=POSITION_POLICIES, default=DEFAULT_POSITION_POLICY)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.selection_ratio < 1.0:
        raise ValueError("selection ratio must be in (0, 1)")
    if min(args.start_index, args.max_examples, args.max_per_task, args.max_target_tokens) < 0:
        raise ValueError("start-index and maximum limits must be non-negative")
    if args.span_neighbor_radius < 0:
        raise ValueError("span-neighbor-radius must be non-negative")

    queries = _read_jsonl(args.queries_jsonl)[args.start_index:]
    if args.max_per_task:
        task_counts: dict[str, int] = defaultdict(int)
        filtered_queries = []
        for query in queries:
            task = str(query.get("task", ""))
            if task_counts[task] >= args.max_per_task:
                continue
            task_counts[task] += 1
            filtered_queries.append(query)
        queries = filtered_queries
    if args.max_examples:
        queries = queries[:args.max_examples]
    contributions = _rows_by_id(args.contribution_jsonl)
    span_oracles = _rows_by_id(args.span_oracle_jsonl)
    named_sidecars = _parse_named_sidecars(args.candidate_jsonl)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = {
        str(row["id"]) for row in _read_jsonl(output)
    } if args.resume and output.exists() else set()
    mode = "a" if completed_ids else "w"

    model, _, processor = from_pretrained(args.checkpoint, device=args.device, dtype=args.dtype)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    fingerprint = model.memory_fingerprint()
    written = 0
    with output.open(mode) as handle:
        for query in queries:
            query_record_id = str(query["id"])
            if query_record_id in completed_ids:
                continue
            memory_id = document_id(query)
            memory = load_latent_memory(Path(args.cache_dir) / memory_id, device=args.device)
            if memory.fingerprint != fingerprint:
                raise ValueError(f"cache {memory_id} was built by an incompatible Encoder")
            k = max(1, math.ceil(args.selection_ratio * memory.num_latents))

            contribution_row = contributions.get(query_record_id)
            contribution_values = (
                contribution_row.get("latent_contributions") if contribution_row else None
            )
            span_row = span_oracles.get(query_record_id)
            span_indices = span_row.get("oracle_latent_indices") if span_row else None
            role_indices = span_row.get("role_latent_indices", {}) if span_row else {}
            external_candidates = {}
            for name, rows in named_sidecars.items():
                candidate_row = rows.get(query_record_id)
                if candidate_row is None:
                    raise ValueError(f"candidate sidecar {name!r} is missing {query_record_id}")
                candidate_document = candidate_row.get("document_id")
                if candidate_document is not None and str(candidate_document) != memory_id:
                    raise ValueError(f"candidate sidecar {name!r} points to a different document")
                candidate_fingerprint = candidate_row.get("cache_fingerprint")
                if candidate_fingerprint not in (None, memory.fingerprint):
                    raise ValueError(f"candidate sidecar {name!r} uses an incompatible cache")
                indices = candidate_row.get(
                    "selected_latent_indices",
                    candidate_row.get("utility_oracle_latent_indices"),
                )
                if not isinstance(indices, list):
                    raise ValueError(f"candidate sidecar {name!r} has no selected indices")
                external_candidates[name] = indices

            candidates = build_candidate_family(
                memory.num_latents,
                k,
                args.selection_ratio,
                args.num_groups,
                contribution_values=contribution_values,
                span_indices=span_indices,
                external_candidates=external_candidates,
                neighbor_radius=args.span_neighbor_radius,
                role_indices=role_indices,
            )
            full_nll, token_count = teacher_forced_nll(
                model,
                processor,
                memory,
                query,
                args.device,
                max_target_tokens=args.max_target_tokens or None,
                answer_separator=args.answer_separator,
                position_policy=args.position_policy,
            )
            null_nll, null_token_count = teacher_forced_nll(
                model,
                processor,
                memory,
                query,
                args.device,
                max_target_tokens=args.max_target_tokens or None,
                answer_separator=args.answer_separator,
                selected_indices=[],
                position_policy=args.position_policy,
            )
            if null_token_count != token_count:
                raise RuntimeError("full and empty memory evaluated different answer-token counts")

            scored_candidates = []
            best_indices: tuple[int, ...] | None = None
            best_nll = math.inf
            best_names: list[str] = []
            for indices, names in candidates.items():
                selected_nll, selected_token_count = teacher_forced_nll(
                    model,
                    processor,
                    memory,
                    query,
                    args.device,
                    max_target_tokens=args.max_target_tokens or None,
                    answer_separator=args.answer_separator,
                    selected_indices=indices,
                    position_policy=args.position_policy,
                )
                if selected_token_count != token_count:
                    raise RuntimeError("candidate evaluated a different answer-token count")
                candidate_gain = full_nll - selected_nll
                scored_candidates.append({
                    "names": names,
                    "candidate_role": _candidate_roles(names),
                    "candidate_latent_indices": list(indices),
                    "k_decoder": len(indices),
                    "answer_nll": selected_nll,
                    "full_nll": full_nll,
                    "null_nll": null_nll,
                    "candidate_gain": candidate_gain,
                    # Exact synergy requires extra singleton/group ablations;
                    # keep the field explicit rather than implying that a
                    # single-latent proxy is an NLL-equivalent interaction term.
                    "candidate_synergy": None,
                })
                if (selected_nll, indices) < (best_nll, best_indices or ()):
                    best_nll = selected_nll
                    best_indices = indices
                    best_names = names
            if best_indices is None:
                raise RuntimeError("utility candidate family is empty")

            metrics = answer_nll_metrics(full_nll, best_nll, null_nll)
            row = {
                "id": query_record_id,
                "document_id": memory_id,
                "task": query.get("task"),
                "utility_oracle_schema_version": UTILITY_ORACLE_SCHEMA_VERSION,
                "utility_oracle_scope": UTILITY_ORACLE_SCOPE,
                "candidate_family": {
                    "balanced_contiguous_group_count": args.num_groups,
                    "group_candidates_require_exact_k": True,
                    "built_in_candidates": [
                        "uniform", "span_oracle_fill", "span_oracle_neighbor",
                        "role fills/unions when supplied",
                    ],
                    "span_neighbor_radius": args.span_neighbor_radius,
                    "role_names": sorted(
                        str(role) for role, indices in (role_indices or {}).items() if indices
                    ),
                    "includes_contribution_topk": contribution_values is not None,
                    "external_candidate_names": sorted(external_candidates),
                },
                "utility_oracle_latent_indices": list(best_indices),
                "utility_oracle_candidate_names": best_names,
                "candidate_answer_nll": scored_candidates,
                "candidate_latent_indices": list(best_indices),
                "candidate_role": _candidate_roles(best_names),
                "candidate_gain": full_nll - best_nll,
                "candidate_synergy": None,
                "candidate_count": len(candidates),
                "selection_ratio": len(best_indices) / memory.num_latents,
                "k_decoder": len(best_indices),
                "full_latent_count": memory.num_latents,
                "num_groups": args.num_groups,
                "position_policy": args.position_policy,
                "teacher_answer_token_count": token_count,
                "cache_fingerprint": memory.fingerprint,
                "cache_schema_version": memory.schema_version,
                **metrics,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
            print(json.dumps({
                "id": query_record_id,
                "candidate_count": len(candidates),
                "best": best_names,
                "answer_nll_retention": metrics["answer_nll_retention"],
            }), flush=True)

    print(json.dumps({
        "out": str(output),
        "records_written": written,
        "records_reused": len(completed_ids),
        "utility_oracle_scope": UTILITY_ORACLE_SCOPE,
        "position_policy": args.position_policy,
        "cache_fingerprint": fingerprint,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
