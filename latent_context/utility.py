"""Pure utility-oracle and answer-NLL metric helpers."""

from __future__ import annotations

from itertools import combinations
import math


def answer_nll_metrics(full_nll: float, selected_nll: float, null_nll: float) -> dict[str, float | None]:
    """Compare a selected memory against full-memory and empty-memory controls.

    Normalized retention is defined only when full memory improves answer NLL
    over empty memory. It is intentionally left unclipped as the primary value
    so regressions (<0) and improvements beyond full (>1) remain visible.
    """
    values = (float(full_nll), float(selected_nll), float(null_nll))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("answer NLL values must be finite")
    full_nll, selected_nll, null_nll = values
    full_gain = null_nll - full_nll
    selected_gain = null_nll - selected_nll
    retention = selected_gain / full_gain if full_gain > 1e-8 else None
    likelihood_ratio = math.exp(max(-80.0, min(80.0, full_nll - selected_nll)))
    return {
        "answer_nll_full": full_nll,
        "answer_nll_selected": selected_nll,
        "answer_nll_null": null_nll,
        "answer_nll_delta_vs_full": selected_nll - full_nll,
        "answer_nll_gain_full_vs_null": full_gain,
        "answer_nll_gain_selected_vs_null": selected_gain,
        "answer_nll_retention": retention,
        "answer_nll_retention_clipped": (
            max(0.0, min(1.0, retention)) if retention is not None else None
        ),
        "answer_likelihood_ratio_vs_full": likelihood_ratio,
    }


def balanced_contiguous_groups(memory_length: int, num_groups: int) -> list[tuple[int, ...]]:
    """Partition ``[0, M)`` into source-ordered groups differing by at most one."""
    if memory_length <= 0:
        raise ValueError("memory_length must be positive")
    if num_groups <= 1 or num_groups > memory_length:
        raise ValueError("num_groups must be in [2, memory_length]")
    base, remainder = divmod(memory_length, num_groups)
    groups = []
    cursor = 0
    for group_index in range(num_groups):
        size = base + int(group_index < remainder)
        groups.append(tuple(range(cursor, cursor + size)))
        cursor += size
    return groups


def exact_group_oracle_candidates(
    memory_length: int,
    selection_ratio: float,
    num_groups: int = 8,
) -> list[tuple[int, ...]]:
    """Enumerate all exact-budget unions of balanced contiguous groups.

    This defines a finite candidate family over which an answer-NLL oracle can
    be exact. It is not presented as the global optimum over all ``M choose K``
    latent subsets.
    """
    if not 0.0 < selection_ratio < 1.0:
        raise ValueError("selection_ratio must be in (0, 1)")
    raw_group_count = selection_ratio * num_groups
    groups_to_select = round(raw_group_count)
    if not math.isclose(raw_group_count, groups_to_select, abs_tol=1e-9):
        raise ValueError("selection_ratio * num_groups must be an integer")
    if not 0 < groups_to_select < num_groups:
        raise ValueError("group oracle must select between one and num_groups - 1 groups")
    target_k = math.ceil(selection_ratio * memory_length)
    groups = balanced_contiguous_groups(memory_length, num_groups)
    candidates = []
    for group_ids in combinations(range(num_groups), groups_to_select):
        indices = tuple(index for group_id in group_ids for index in groups[group_id])
        if len(indices) == target_k:
            candidates.append(indices)
    if not candidates:
        raise ValueError("balanced groups produced no exact-K oracle candidates")
    return candidates
