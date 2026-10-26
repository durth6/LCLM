"""Query-aware selection over cached LCLM latent memories.

The cache in this module deliberately contains only continuous latent vectors and
their structural metadata.  It never stores source text or source token ids, so
online selection cannot silently fall back to text retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


_CACHE_TENSORS = (
    "latents",
    "latent_start",
    "latent_end",
    "latent_index",
    "segment_id",
    "window_id",
)
CACHE_SCHEMA_VERSION = 1


@dataclass
class LatentMemory:
    """A reusable, pre-Adapter memory for one source context.

    Shapes are ``latents=[M, D_E]`` and all structural tensors ``[M]``.  Source
    text and source token ids are intentionally not accepted by this type.
    """

    latents: torch.Tensor
    latent_start: torch.Tensor
    latent_end: torch.Tensor
    latent_index: torch.Tensor
    segment_id: torch.Tensor
    window_id: torch.Tensor
    fingerprint: str
    source_id: Optional[str] = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        self.validate()

    @property
    def num_latents(self) -> int:
        return int(self.latents.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.latents.shape[1])

    def validate(self) -> None:
        if self.latents.ndim != 2:
            raise ValueError(f"latents must have shape [M, D], got {tuple(self.latents.shape)}")
        expected = self.latents.shape[0]
        for name in _CACHE_TENSORS[1:]:
            value = getattr(self, name)
            if value.ndim != 1 or value.shape[0] != expected:
                raise ValueError(
                    f"{name} must have shape [{expected}], got {tuple(value.shape)}"
                )
        if expected == 0:
            raise ValueError("LatentMemory cannot be empty")
        if expected > 1 and torch.any(self.latent_index[1:] <= self.latent_index[:-1]):
            raise ValueError("latent_index must be strictly increasing in source order")
        if torch.any(self.latent_end < self.latent_start):
            raise ValueError("latent_end must be >= latent_start")

    def to(self, device: torch.device | str) -> "LatentMemory":
        return LatentMemory(
            latents=self.latents.to(device),
            latent_start=self.latent_start.to(device),
            latent_end=self.latent_end.to(device),
            latent_index=self.latent_index.to(device),
            segment_id=self.segment_id.to(device),
            window_id=self.window_id.to(device),
            fingerprint=self.fingerprint,
            source_id=self.source_id,
            schema_version=self.schema_version,
        )

    def select(self, indices: torch.Tensor) -> "LatentMemory":
        """Return a source-order-preserving sub-memory.

        ``indices`` must already be sorted by original latent position.  This
        catches a common error where Top-K score order is fed to the decoder.
        """
        if indices.ndim != 1:
            raise ValueError("indices must have shape [K]")
        if indices.numel() == 0:
            raise ValueError("at least one latent must be selected")
        if not torch.equal(indices, torch.sort(indices).values):
            raise ValueError("selected indices must be restored to source order")
        if int(indices.min()) < 0 or int(indices.max()) >= self.num_latents:
            raise IndexError("selected latent index is out of range")
        local_indices = indices.to(self.latents.device)
        return LatentMemory(
            latents=self.latents.index_select(0, local_indices),
            latent_start=self.latent_start.index_select(0, local_indices),
            latent_end=self.latent_end.index_select(0, local_indices),
            latent_index=self.latent_index.index_select(0, local_indices),
            segment_id=self.segment_id.index_select(0, local_indices),
            window_id=self.window_id.index_select(0, local_indices),
            fingerprint=self.fingerprint,
            source_id=self.source_id,
            schema_version=self.schema_version,
        )


@dataclass
class SelectionOutput:
    """Top-K result. ``indices`` are always in original source order."""

    indices: torch.Tensor  # [B, K]
    scores: torch.Tensor  # [B, M]
    selected_count: torch.Tensor  # [B]


@dataclass
class DynamicSelectionOutput:
    """One-row result from the discrete dynamic-budget selector.

    Dynamic budgets cannot be represented by the fixed ``[B, K]`` shape used by
    :class:`SelectionOutput`, so this result keeps the selected row together
    with the chosen ratio and the auditable budget sweep.
    """

    indices: torch.Tensor  # [K], source ordered
    scores: torch.Tensor  # [M]
    selected_count: int
    selected_ratio: float
    predicted_coverage: float
    budget_sweep: list[dict[str, Any]]
    # Coverage of the final (possibly coverage-reranked) selected set.  This
    # is kept separate from ``predicted_coverage`` because budget decisions
    # must be based on relevance-only Top-K predictions.
    selected_coverage: float | None = None


def make_memory_fingerprint(payload: Dict[str, Any]) -> str:
    """Create a stable fingerprint for model/tokenizer/cache compatibility."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def save_latent_memory(memory: LatentMemory, cache_dir: str | Path) -> None:
    """Persist a cache as Safetensors plus text-only metadata.

    The metadata file intentionally has no source-text field.  This is an
    enforceable part of the latent-only retrieval contract.
    """
    from safetensors.torch import save_file

    if memory.schema_version != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported latent cache schema {memory.schema_version}; expected {CACHE_SCHEMA_VERSION}"
        )
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    tensors = {name: getattr(memory, name).detach().cpu().contiguous() for name in _CACHE_TENSORS}
    save_file(tensors, str(cache_path / "memory.safetensors"))
    metadata = {
        "schema_version": memory.schema_version,
        "fingerprint": memory.fingerprint,
        "source_id": memory.source_id,
        "num_latents": memory.num_latents,
        "embedding_dim": memory.embedding_dim,
    }
    (cache_path / "memory_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_latent_memory(cache_dir: str | Path, device: Optional[str | torch.device] = None) -> LatentMemory:
    """Load a cache written by :func:`save_latent_memory`."""
    from safetensors.torch import load_file

    cache_path = Path(cache_dir)
    metadata = json.loads((cache_path / "memory_metadata.json").read_text())
    schema_version = int(metadata.get("schema_version", 1))
    if schema_version != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported latent cache schema {schema_version}; expected {CACHE_SCHEMA_VERSION}"
        )
    tensors = load_file(str(cache_path / "memory.safetensors"), device=str(device or "cpu"))
    return LatentMemory(
        **{name: tensors[name] for name in _CACHE_TENSORS},
        fingerprint=metadata["fingerprint"],
        source_id=metadata.get("source_id"),
        schema_version=schema_version,
    )


def select_topk_in_source_order(scores: torch.Tensor, k: int) -> SelectionOutput:
    """Select score Top-K, then restore original latent order.

    A stable score sort makes ties deterministic: earlier source positions win.
    Decoder order is independently restored by sorting the selected indices.
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape [B, M], got {tuple(scores.shape)}")
    if k <= 0:
        raise ValueError("k must be positive")
    batch_size, memory_length = scores.shape
    if memory_length == 0:
        raise ValueError("cannot select from an empty memory")
    actual_k = min(k, memory_length)
    ranked = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :actual_k]
    indices = torch.sort(ranked, dim=-1).values
    return SelectionOutput(
        indices=indices,
        scores=scores,
        selected_count=torch.full((batch_size,), actual_k, dtype=torch.int32, device=scores.device),
    )


def select_uniform_indices(memory_length: int, k: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Choose ``k`` source-strided latent indices, including both endpoints."""
    if memory_length <= 0:
        raise ValueError("memory_length must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    actual_k = min(k, memory_length)
    if actual_k == 1:
        return torch.zeros(1, dtype=torch.long, device=device)
    # Integer arithmetic avoids float rounding duplicates and remains exactly
    # source ordered for every M/K pair.
    positions = torch.arange(actual_k, dtype=torch.long, device=device)
    return positions * (memory_length - 1) // (actual_k - 1)


def select_random_indices(
    memory_length: int,
    k: int,
    generator: Optional[torch.Generator] = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Choose a reproducible random-K source-order baseline."""
    if memory_length <= 0:
        raise ValueError("memory_length must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    actual_k = min(k, memory_length)
    sampled = torch.randperm(memory_length, generator=generator, device=device)[:actual_k]
    return torch.sort(sampled).values


def expand_with_neighbors(
    ranked_indices: torch.Tensor,
    memory_length: int,
    k: int,
    radius: int,
) -> torch.Tensor:
    """Expand high-score seed latents into local source windows under budget ``K``.

    ``ranked_indices`` is in descending relevance order, unlike the indices
    passed to the Decoder.  We first allocate whole ``2 * radius + 1`` windows
    around the top seeds (without exceeding the total decoder budget), then
    use the remaining score-ranked candidates as deterministic filler.  The
    returned indices are unique, exactly ``min(K, M)`` long, and restored to
    source order.  This makes a neighbour-expansion ablation directly
    comparable to ordinary Top-K in both KV-cache and prefill cost.
    """
    if ranked_indices.ndim != 1:
        raise ValueError("ranked_indices must have shape [M]")
    if memory_length <= 0:
        raise ValueError("memory_length must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if ranked_indices.numel() != memory_length:
        raise ValueError("ranked_indices must contain every latent exactly once")
    if torch.unique(ranked_indices).numel() != memory_length or (
        int(ranked_indices.min()) < 0 or int(ranked_indices.max()) >= memory_length
    ):
        raise ValueError("ranked_indices must be a permutation of [0, M)")

    actual_k = min(k, memory_length)
    if radius == 0 or actual_k == memory_length:
        return torch.sort(ranked_indices[:actual_k]).values

    selected: set[int] = set()
    # Reserving a full nominal window per seed guarantees that expansion never
    # consumes more than K.  Boundary windows or overlaps simply leave budget
    # for score-ranked fillers below.
    seed_budget = max(1, actual_k // (2 * radius + 1))
    for seed in ranked_indices[:seed_budget].tolist():
        for index in range(max(0, seed - radius), min(memory_length, seed + radius + 1)):
            selected.add(index)
    for index in ranked_indices.tolist():
        if len(selected) >= actual_k:
            break
        selected.add(index)
    return torch.tensor(sorted(selected), dtype=torch.long, device=ranked_indices.device)


# ---------------------------------------------------------------------------
# Multi-slot / sub-query selection (query-side decomposition)
#
# These helpers decompose one selector question into several query "slots"
# (sub-queries or content tokens), score the memory once per slot, and merge
# the per-slot rankings into a single exact-K selection.  All slot text is
# derived from the question itself at answer time; no oracle labels, no
# answer NLL, and no source context are involved.
# ---------------------------------------------------------------------------

_CHAT_USER_MARKER = "<|im_start|>user\n"
_ENTITY_ENUMERATION = re.compile(
    r"\bfor\s+([A-Za-z0-9][A-Za-z0-9_-]*(?:\s*,\s*[A-Za-z0-9][A-Za-z0-9_-]*)+"
    r"(?:,?\s+and\s+[A-Za-z0-9][A-Za-z0-9_-]*)?)\s+(?:mentioned|appearing|listed)",
    re.IGNORECASE,
)
_ENTITY_SPLIT = re.compile(r",\s*|\s+and\s+", re.IGNORECASE)


def user_question_text(selector_query: str) -> str:
    """Return the user-turn text of a chat-templated selector query.

    Slot construction operates on the question itself.  Stripping the system
    prompt and generation suffix keeps slot budgets from being spent on
    boilerplate template tokens; queries without the template are returned
    unchanged.
    """
    if not isinstance(selector_query, str) or not selector_query.strip():
        raise ValueError("selector_query must be a non-empty string")
    if _CHAT_USER_MARKER in selector_query:
        user = selector_query.split(_CHAT_USER_MARKER, 1)[1]
        stop = user.find("<|im_end|>")
        if stop >= 0:
            user = user[:stop]
        user = user.strip()
        if user:
            return user
    return selector_query.strip()


def extract_entity_slots(selector_query: str) -> list[str]:
    """Rewrite an enumerated question into one sub-query per enumerated entity.

    Multi-entity questions such as ``... for a, b, c, and d mentioned ...``
    are split into one full-query slot plus one single-entity sub-query per
    entity, so each slot can match its own evidence instead of averaging all
    of them into one pooled vector.  Questions without an enumeration return
    only the original query.  The enumeration grammar is derived purely from
    the question text at answer time.
    """
    question = user_question_text(selector_query)
    match = _ENTITY_ENUMERATION.search(question)
    if match is None:
        return [selector_query]
    entities = []
    for raw_entity in _ENTITY_SPLIT.split(match.group(1)):
        entity = raw_entity.strip()
        # The optional comma of ", and" is consumed by the comma alternative,
        # so a trailing list item can still start with a bare "and".
        entity = re.sub(r"^and\s+", "", entity, flags=re.IGNORECASE)
        if entity:
            entities.append(entity)
    if len(entities) < 2:
        return [selector_query]
    start, end = match.span(1)
    slots = [selector_query]
    for entity in entities:
        slots.append(question[:start] + entity + question[end:])
    return slots


def split_question_sentences(selector_query: str, max_slots: int) -> list[str]:
    """Split a selector query into sentence-level slots (original query first)."""
    if max_slots <= 0:
        raise ValueError("max_slots must be positive")
    question = user_question_text(selector_query)
    pieces = re.split(r"(?<=[.?!])\s+|\n+", question)
    sentences = [piece.strip() for piece in pieces if len(piece.strip()) >= 8]
    if not sentences:
        return [selector_query]
    slots = [selector_query] + sentences[-max_slots:]
    return slots


def select_slots_union(slot_scores: torch.Tensor, k: int) -> torch.Tensor:
    """Merge per-slot scores into one exact-K, source-ordered selection.

    Each of the ``G`` slots first claims ``ceil(K / G)`` of its best latents;
    the deduplicated union is pruned back to ``K`` by the slots' mean score
    (consensus) when it overflows and filled to ``K`` by the slots' max score
    (any-slot relevance) when it underflows.  ``G=1`` therefore reproduces
    plain relevance Top-K, which keeps the ablation directly comparable.
    """
    if slot_scores.ndim != 2 or slot_scores.shape[0] == 0:
        raise ValueError(f"slot_scores must have shape [G, M], got {tuple(slot_scores.shape)}")
    if not bool(torch.isfinite(slot_scores).all()):
        raise ValueError("slot_scores must be finite")
    if k <= 0:
        raise ValueError("k must be positive")
    slot_count, memory_length = slot_scores.shape
    actual_k = min(int(k), memory_length)
    per_slot = max(1, math.ceil(actual_k / slot_count))
    claimed = torch.argsort(slot_scores, dim=-1, descending=True, stable=True)[:, :per_slot]
    union = torch.unique(claimed.reshape(-1))
    mean_scores = slot_scores.mean(dim=0)
    max_scores = slot_scores.max(dim=0).values
    if int(union.numel()) > actual_k:
        keep = torch.argsort(mean_scores[union], descending=True, stable=True)[:actual_k]
        union = union[torch.sort(keep).values]
    if int(union.numel()) < actual_k:
        selected_mask = torch.zeros(memory_length, dtype=torch.bool, device=slot_scores.device)
        selected_mask[union] = True
        fill_order = torch.argsort(max_scores, descending=True, stable=True)
        deficit = actual_k - int(union.numel())
        filler = fill_order[~selected_mask[fill_order]][:deficit]
        union = torch.sort(torch.cat((union, filler))).values
    return union.to(dtype=torch.long)


def reciprocal_rank_fusion(slot_scores: torch.Tensor, fusion_k: int = 60) -> torch.Tensor:
    """Fuse per-slot rankings with standard RRF into one [M] score vector."""
    if slot_scores.ndim != 2 or slot_scores.shape[0] == 0:
        raise ValueError(f"slot_scores must have shape [G, M], got {tuple(slot_scores.shape)}")
    if fusion_k <= 0:
        raise ValueError("fusion_k must be positive")
    ranks = torch.argsort(torch.argsort(slot_scores, dim=-1, descending=True, stable=True), dim=-1)
    return (1.0 / (float(fusion_k) + ranks.to(torch.float32))).sum(dim=0)


def content_token_slots(
    token_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    token_texts: Sequence[str],
    max_slots: int,
) -> torch.Tensor:
    """Pick the most distinctive question tokens as retrieval slots.

    Candidates are non-template alphanumeric tokens outside a small stopword
    set.  Distinctiveness is measured as cosine distance from the mean
    candidate-token embedding, a cheap IDF proxy that needs no external
    corpus.  The returned slots are the raw token hidden states, so they can
    be projected by the trained selector exactly like pooled query vectors.
    """
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
        "is", "it", "me", "my", "of", "on", "or", "the", "this", "to", "will",
        "you", "your", "what", "which", "that", "with", "i", "text", "given",
    }
    if token_hidden.ndim != 2 or attention_mask.shape != token_hidden.shape[:1]:
        raise ValueError("token_hidden must be [T, D] with a matching attention_mask")
    if len(token_texts) != token_hidden.shape[0]:
        raise ValueError("token_texts must provide one string per token")
    if max_slots <= 0:
        raise ValueError("max_slots must be positive")
    if token_hidden.shape[0] == 0:
        raise ValueError("token_hidden cannot be empty")
    mask = attention_mask.to(dtype=torch.bool, device=token_hidden.device)
    candidate_indices: list[int] = []
    for index in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        token = str(token_texts[index]).strip().lower()
        # RULER-style entities are hyphenated ("innocent-clause"), so internal
        # hyphens stay admissible; punctuation and stopwords do not.
        if len(token) < 2 or not all(ch.isalnum() or ch == "-" for ch in token):
            continue
        if token in stopwords:
            continue
        candidate_indices.append(index)
    if not candidate_indices:
        return token_hidden[:1]
    embeddings = token_hidden[candidate_indices]
    if embeddings.shape[0] <= max_slots:
        return embeddings
    center = F.normalize(embeddings.mean(dim=0), dim=-1)
    norms = F.normalize(embeddings, dim=-1)
    distinctiveness = 1.0 - norms @ center
    order = torch.argsort(distinctiveness, descending=True)[:max_slots]
    return embeddings[order]


def contiguous_coverage_groups(
    memory_length: int,
    num_groups: int = 8,
    prefix: str = "position",
) -> dict[str, tuple[int, ...]]:
    """Create query-independent source-position coverage groups.

    These groups are deliberately structural: they use only the cached latent
    count and source order, never answer/oracle labels.  Callers may instead
    provide semantic role groups (for example offline diagnostics) through the
    ``groups`` argument of :func:`coverage_aware_greedy`.
    """
    if memory_length <= 0:
        raise ValueError("memory_length must be positive")
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    actual_groups = min(int(num_groups), memory_length)
    base, remainder = divmod(memory_length, actual_groups)
    groups: dict[str, tuple[int, ...]] = {}
    cursor = 0
    for group_index in range(actual_groups):
        size = base + int(group_index < remainder)
        groups[f"{prefix}_{group_index:02d}"] = tuple(range(cursor, cursor + size))
        cursor += size
    return groups


def memory_structure_groups(
    memory: LatentMemory,
    *,
    fallback_groups: int = 8,
    prefix: str = "window",
) -> dict[str, tuple[int, ...]]:
    """Build source-structure groups from cached segment/window metadata.

    Window/segment boundaries are available online in the latent cache and are
    therefore safe for prediction-time coverage.  A single-window cache uses
    equal source buckets so the reranker still has a useful, deterministic
    spread signal.
    """
    if not isinstance(memory, LatentMemory):
        raise TypeError("memory must be a LatentMemory")
    keys = [
        (int(segment), int(window))
        for segment, window in zip(memory.segment_id.tolist(), memory.window_id.tolist())
    ]
    unique_keys = list(dict.fromkeys(keys))
    if len(unique_keys) <= 1:
        return contiguous_coverage_groups(memory.num_latents, fallback_groups, prefix=prefix)
    groups: dict[str, list[int]] = {f"{prefix}_{index:02d}": [] for index in range(len(unique_keys))}
    key_to_name = {key: f"{prefix}_{index:02d}" for index, key in enumerate(unique_keys)}
    for index, key in enumerate(keys):
        groups[key_to_name[key]].append(index)
    return {name: tuple(indices) for name, indices in groups.items() if indices}


def coverage_for_selection(
    selected_indices: Sequence[int] | torch.Tensor,
    groups: Mapping[str, Sequence[int]],
) -> dict[str, float | int | list[str]]:
    """Return auditable role/group coverage for a source-ordered selection."""
    selected = {int(index) for index in torch.as_tensor(selected_indices).tolist()}
    covered = sorted(
        str(role) for role, indices in groups.items()
        if selected.intersection(int(index) for index in indices)
    )
    total = len(groups)
    return {
        "covered_groups": covered,
        "covered_group_count": len(covered),
        "total_group_count": total,
        "coverage": len(covered) / total if total else 1.0,
    }


def source_span_summary(
    memory: LatentMemory,
    selected_indices: Sequence[int] | torch.Tensor,
) -> dict[str, float | int]:
    """Summarize source-token spans represented by a selected latent set.

    The summary is structural metadata from the latent cache, so it is safe to
    record on the answer path without re-reading source text.  ``indices`` are
    local latent positions and must already be in source order.  A source run
    continues while the next span touches or overlaps the covered extent;
    gaps between spans start a new run.
    """
    if not isinstance(memory, LatentMemory):
        raise TypeError("memory must be a LatentMemory")
    indices = torch.as_tensor(selected_indices, dtype=torch.long, device=memory.latent_start.device)
    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError("selected_indices must be a non-empty 1-D sequence")
    if not torch.equal(indices, torch.sort(indices).values):
        raise ValueError("selected_indices must be restored to source order")
    if int(indices.min()) < 0 or int(indices.max()) >= memory.num_latents:
        raise IndexError("selected latent index is out of range")

    starts = memory.latent_start.index_select(0, indices).to(dtype=torch.long)
    ends = memory.latent_end.index_select(0, indices).to(dtype=torch.long)
    widths = (ends - starts).to(dtype=torch.float32)
    start_values = starts.tolist()
    end_values = ends.tolist()
    run_count = 1
    covered_end = int(end_values[0])
    for start, end in zip(start_values[1:], end_values[1:]):
        if int(start) > covered_end:
            run_count += 1
        covered_end = max(covered_end, int(end))
    start_min = int(starts.min().item())
    end_max = int(ends.max().item())
    return {
        "selected_source_start_min": start_min,
        "selected_source_end_max": end_max,
        "selected_source_span_mean": float(widths.mean().item()),
        "selected_source_token_extent": end_max - start_min,
        "selected_source_run_count": run_count,
    }


def coverage_aware_greedy(
    scores: torch.Tensor,
    k: int,
    groups: Mapping[str, Sequence[int]] | None = None,
    *,
    neighbor_radius: int = 0,
    relevance_weight: float = 1.0,
    coverage_weight: float = 0.25,
    neighbor_weight: float = 0.0,
) -> torch.Tensor:
    """Select exactly ``K`` latents with auditable role/group coverage.

    The greedy objective combines stable rank relevance with a bonus for a
    previously uncovered group and, optionally, local source continuity.  The
    result is always restored to source order before it is returned.  The
    default weights keep relevance dominant; callers opt into this reranker
    explicitly so legacy relevance-only behavior is unchanged.
    """
    if scores.ndim != 1:
        raise ValueError("scores must have shape [M]")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be finite")
    if k <= 0:
        raise ValueError("k must be positive")
    if neighbor_radius < 0:
        raise ValueError("neighbor_radius must be non-negative")
    if relevance_weight < 0 or coverage_weight < 0 or neighbor_weight < 0:
        raise ValueError("reranker weights must be non-negative")
    memory_length = int(scores.numel())
    if memory_length == 0:
        raise ValueError("cannot select from an empty memory")
    actual_k = min(int(k), memory_length)
    if groups is None:
        groups = contiguous_coverage_groups(memory_length)
    normalized_groups: dict[str, set[int]] = {}
    for role, raw_indices in groups.items():
        indices = {int(index) for index in raw_indices}
        if any(index < 0 or index >= memory_length for index in indices):
            raise IndexError(f"coverage group {role!r} contains an out-of-range index")
        if indices:
            normalized_groups[str(role)] = indices

    ranked = torch.argsort(scores, descending=True, stable=True)
    # Rank scores avoid making the coverage coefficient depend on the scale of
    # a particular checkpoint or selector head.
    relevance = torch.zeros_like(scores, dtype=torch.float32)
    if memory_length == 1:
        relevance[0] = 1.0
    else:
        relevance[ranked] = torch.linspace(
            1.0, 0.0, memory_length, device=scores.device, dtype=torch.float32
        )
    # Materialize group membership once and keep the greedy loop on the target
    # device.  The previous scalar Python loop synchronized a CUDA tensor for
    # every candidate, which is prohibitive for 32K contexts.
    role_names = list(normalized_groups)
    membership = torch.zeros(
        (len(role_names), memory_length), dtype=torch.bool, device=scores.device
    )
    for role_index, role in enumerate(role_names):
        indices = torch.tensor(
            sorted(normalized_groups[role]), dtype=torch.long, device=scores.device
        )
        membership[role_index, indices] = True

    selected_mask = torch.zeros(memory_length, dtype=torch.bool, device=scores.device)
    covered_mask = torch.zeros(len(role_names), dtype=torch.bool, device=scores.device)
    selected: list[torch.Tensor] = []

    # Pick only enough anchors to cover the available structural/role groups.
    # Once every group is represented, the rest of K is a single relevance
    # (optionally neighbor-aware) Top-K fill rather than another M-sized loop.
    anchor_steps = min(actual_k, len(role_names)) if coverage_weight > 0 else 0
    for _ in range(anchor_steps):
        new_role_count = (
            ((~covered_mask).to(membership.dtype).unsqueeze(1) * membership)
            .sum(dim=0)
            .to(torch.float32)
            if role_names else torch.zeros(memory_length, dtype=torch.float32, device=scores.device)
        )
        utility = relevance_weight * relevance + coverage_weight * new_role_count
        has_new_role = new_role_count > 0
        if bool(has_new_role.any()):
            utility = utility.masked_fill(~has_new_role, -torch.inf)
        utility = utility.masked_fill(selected_mask, -torch.inf)
        # torch.argmax returns the first source index on an exact tie, matching
        # the stable Top-K policy used by ``select_topk_in_source_order``.
        best_index = torch.argmax(utility)
        selected.append(best_index)
        selected_mask[best_index] = True
        if role_names:
            covered_mask = covered_mask | membership[:, best_index]

        if bool(covered_mask.all()):
            break

    remaining = actual_k - len(selected)
    if remaining > 0:
        utility = relevance_weight * relevance
        if neighbor_radius and neighbor_weight and selected:
            neighbor_kernel = torch.ones(
                1, 1, 2 * neighbor_radius + 1,
                dtype=torch.float32,
                device=scores.device,
            )
            selected_values = selected_mask.to(torch.float32).view(1, 1, -1)
            neighbor_count = F.conv1d(
                F.pad(selected_values, (neighbor_radius, neighbor_radius)), neighbor_kernel
            ).view(-1)
            utility = utility + neighbor_weight * neighbor_count
        utility = utility.masked_fill(selected_mask, -torch.inf)
        filler = torch.argsort(utility, descending=True, stable=True)[:remaining]
        selected.extend(filler.unbind())
    return torch.sort(torch.stack(selected).to(dtype=torch.long)).values


def select_dynamic_budget(
    scores: torch.Tensor,
    memory_length: int | None = None,
    *,
    minimum_ratio: float = 0.25,
    budget_ratios: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    groups: Mapping[str, Sequence[int]] | None = None,
    coverage_threshold: float = 0.75,
    coverage_weight: float = 0.25,
) -> DynamicSelectionOutput:
    """Choose the smallest discrete budget whose predicted group coverage passes.

    This is a prediction-only policy.  It never consumes answer/NLL/oracle
    labels; the supplied groups are structural or selector-produced metadata.
    ``budget_sweep`` is retained in the result so evaluation can report a
    cost-quality Pareto curve instead of only the final accuracy.
    """
    if scores.ndim != 1:
        raise ValueError("scores must have shape [M]")
    actual_memory_length = int(scores.numel())
    if memory_length is not None and int(memory_length) != actual_memory_length:
        raise ValueError("memory_length does not match scores")
    if not 0.0 < minimum_ratio <= 1.0:
        raise ValueError("minimum_ratio must be in (0, 1]")
    if not 0.0 <= coverage_threshold <= 1.0:
        raise ValueError("coverage_threshold must be in [0, 1]")
    if coverage_weight < 0:
        raise ValueError("coverage_weight must be non-negative")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be finite")
    ratios = sorted({float(ratio) for ratio in budget_ratios})
    if not ratios or any(not 0.0 < ratio <= 1.0 for ratio in ratios):
        raise ValueError("budget_ratios must contain values in (0, 1]")
    ratios = [ratio for ratio in ratios if ratio >= minimum_ratio]
    if not ratios:
        ratios = [1.0]
    if groups is None:
        groups = contiguous_coverage_groups(actual_memory_length)
    sweep: list[dict[str, float | int]] = []
    chosen_ratio_k: tuple[float, int, float] | None = None
    for ratio in ratios:
        k = max(1, min(actual_memory_length, math.ceil(ratio * actual_memory_length)))
        # The sweep is deliberately relevance-only.  Running the coverage
        # reranker at each point would make the predicted coverage depend on
        # the diagnostic objective itself and can select an artificially small
        # budget.  The final chosen set is reranked below after the budget is
        # decided.
        relevance_indices = torch.sort(
            torch.argsort(scores, descending=True, stable=True)[:k]
        ).values
        predicted_coverage = float(
            coverage_for_selection(relevance_indices, groups)["coverage"]
        )
        sweep.append({"ratio": ratio, "k": k, "predicted_coverage": predicted_coverage})
        if chosen_ratio_k is None and predicted_coverage >= coverage_threshold:
            chosen_ratio_k = (ratio, k, predicted_coverage)
    if chosen_ratio_k is None:
        last = sweep[-1]
        chosen_ratio_k = (
            float(last["ratio"]),
            int(last["k"]),
            float(last["predicted_coverage"]),
        )
    ratio, k, predicted_coverage = chosen_ratio_k
    indices = coverage_aware_greedy(
        scores,
        k,
        groups,
        coverage_weight=coverage_weight,
    )
    selected_coverage = float(coverage_for_selection(indices, groups)["coverage"])
    return DynamicSelectionOutput(
        indices=indices,
        scores=scores,
        selected_count=int(indices.numel()),
        selected_ratio=ratio,
        predicted_coverage=predicted_coverage,
        budget_sweep=sweep,
        selected_coverage=selected_coverage,
    )


class CosineSelector(nn.Module):
    """Training-free cosine/dot-product baseline over pre-Adapter latents."""

    def __init__(self, normalize: bool = True) -> None:
        super().__init__()
        self.normalize = normalize

    def score(self, question_vectors: torch.Tensor, memory: LatentMemory) -> torch.Tensor:
        if question_vectors.ndim != 2:
            raise ValueError("question_vectors must have shape [B, D]")
        if question_vectors.shape[-1] != memory.embedding_dim:
            raise ValueError(
                f"question dim {question_vectors.shape[-1]} != latent dim {memory.embedding_dim}"
            )
        latents = memory.latents.to(question_vectors.device, dtype=question_vectors.dtype)
        if self.normalize:
            question_vectors = F.normalize(question_vectors, dim=-1)
            latents = F.normalize(latents, dim=-1)
        return question_vectors @ latents.T

    def forward(self, question_vectors: torch.Tensor, memory: LatentMemory, k: int) -> SelectionOutput:
        return select_topk_in_source_order(self.score(question_vectors, memory), k)


class BiEncoderSelector(nn.Module):
    """Low-dimensional trainable selector for the Stage-A/Stage-B MVP."""

    def __init__(self, input_dim: int, retrieval_dim: int = 128, normalize: bool = True) -> None:
        super().__init__()
        self.question_projection = nn.Linear(input_dim, retrieval_dim, bias=False)
        self.latent_projection = nn.Linear(input_dim, retrieval_dim, bias=False)
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.normalize = normalize

    def score(self, question_vectors: torch.Tensor, memory: LatentMemory) -> torch.Tensor:
        # The released LCLM Encoder normally runs in BF16 while a freshly
        # constructed retrieval head is FP32.  Keep the small selector in its
        # parameter dtype for stable Stage-A optimization instead of requiring
        # callers to mutate the frozen Encoder's dtype.
        selector_dtype = self.question_projection.weight.dtype
        q = self.question_projection(question_vectors.to(dtype=selector_dtype))
        z = self.latent_projection(memory.latents.to(q.device, dtype=selector_dtype))
        if self.normalize:
            q = F.normalize(q, dim=-1)
            z = F.normalize(z, dim=-1)
        temperature = self.log_temperature.exp().clamp(min=0.01, max=100.0)
        return (q @ z.T) / temperature

    def forward(self, question_vectors: torch.Tensor, memory: LatentMemory, k: int) -> SelectionOutput:
        return select_topk_in_source_order(self.score(question_vectors, memory), k)

    def score_slots(self, slot_vectors: torch.Tensor, memory: LatentMemory) -> torch.Tensor:
        """Score the memory against several query slots with one latent projection.

        The slot vectors reuse the same question projection, normalization, and
        temperature as :meth:`score`, so a single-slot call reproduces the
        pooled-vector score exactly.  Returns ``[G, M]``.
        """
        if slot_vectors.ndim != 2:
            raise ValueError(f"slot_vectors must have shape [G, D], got {tuple(slot_vectors.shape)}")
        selector_dtype = self.question_projection.weight.dtype
        q = self.question_projection(slot_vectors.to(dtype=selector_dtype))
        z = self.latent_projection(memory.latents.to(q.device, dtype=selector_dtype))
        if self.normalize:
            q = F.normalize(q, dim=-1)
            z = F.normalize(z, dim=-1)
        temperature = self.log_temperature.exp().clamp(min=0.01, max=100.0)
        return (q @ z.T) / temperature


class PositionConditioner(nn.Module):
    """Feature-only position/gap conditioning for selected Decoder latents.

    The final projection is zero-initialized, preserving base-LCLM behavior at
    initialization while making the positional branch trainable in Stage B.
    """

    def __init__(self, decoder_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, decoder_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, embeddings: torch.Tensor, memory: LatentMemory, source_length: int) -> torch.Tensor:
        if embeddings.ndim != 2 or embeddings.shape[0] != memory.num_latents:
            raise ValueError("embeddings must have shape [K, D] matching selected memory")
        if source_length <= 0:
            raise ValueError("source_length must be positive")
        device = embeddings.device
        start = memory.latent_start.to(device=device, dtype=embeddings.dtype) / source_length
        end = memory.latent_end.to(device=device, dtype=embeddings.dtype) / source_length
        index = memory.latent_index.to(device=device, dtype=embeddings.dtype)
        gaps = torch.zeros_like(index)
        if memory.num_latents > 1:
            gaps[1:] = (index[1:] - index[:-1] - 1).clamp_min(0) / max(memory.num_latents, 1)
        run_start = torch.ones_like(index)
        if memory.num_latents > 1:
            run_start[1:] = (index[1:] != index[:-1] + 1).to(embeddings.dtype)
        features = torch.stack((start, end, gaps, run_start), dim=-1)
        return embeddings + self.net(features).to(embeddings.dtype)


def save_selector_checkpoint(
    checkpoint_dir: str | Path,
    selector: BiEncoderSelector,
    positioner: Optional[PositionConditioner],
    base_memory_fingerprint: str,
    training_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist learned selector state as Safetensors plus explicit metadata.

    Question-encoder weights are intentionally absent: the MVP uses the frozen
    LCLM encoder backbone, whose exact identity is already bound by
    ``base_memory_fingerprint``.  Keeping this checkpoint small also makes it
    practical to iterate Stage A without duplicating multi-GB model weights.
    ``training_metadata`` is merged into the JSON metadata without changing the
    loader schema, so contribution-distilled checkpoints remain loadable by
    existing inference code.
    """
    from safetensors.torch import save_file

    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    tensors = {
        f"selector.{name}": tensor.detach().cpu().contiguous()
        for name, tensor in selector.state_dict().items()
    }
    if positioner is not None:
        tensors.update({
            f"positioner.{name}": tensor.detach().cpu().contiguous()
            for name, tensor in positioner.state_dict().items()
        })
    save_file(tensors, str(path / "selector.safetensors"))
    metadata = {
        "schema_version": 1,
        "selector_type": "bi_encoder",
        "input_dim": selector.question_projection.in_features,
        "retrieval_dim": selector.question_projection.out_features,
        "normalize": selector.normalize,
        "position_mode": "feature_only" if positioner is not None else "none",
        "position_decoder_dim": positioner.net[-1].out_features if positioner is not None else None,
        "position_hidden_dim": positioner.net[0].out_features if positioner is not None else None,
        "base_memory_fingerprint": base_memory_fingerprint,
    }
    if training_metadata:
        # Keep the checkpoint schema backward compatible while recording the
        # exact auxiliary-target definition used to train this selector.
        metadata.update(training_metadata)
    (path / "selector_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_selector_checkpoint(
    checkpoint_dir: str | Path,
    expected_memory_fingerprint: str,
) -> tuple[BiEncoderSelector, Optional[PositionConditioner], Dict[str, Any]]:
    """Load a learned selector and reject incompatible latent memories."""
    from safetensors.torch import load_file

    path = Path(checkpoint_dir)
    metadata = json.loads((path / "selector_metadata.json").read_text())
    if metadata.get("schema_version") != 1 or metadata.get("selector_type") != "bi_encoder":
        raise ValueError("unsupported selector checkpoint schema")
    if metadata.get("base_memory_fingerprint") != expected_memory_fingerprint:
        raise ValueError("selector checkpoint was trained for a different latent-memory encoder")
    selector = BiEncoderSelector(
        input_dim=int(metadata["input_dim"]),
        retrieval_dim=int(metadata["retrieval_dim"]),
        normalize=bool(metadata.get("normalize", True)),
    )
    positioner = None
    if metadata.get("position_mode") == "feature_only":
        positioner = PositionConditioner(
            decoder_dim=int(metadata["position_decoder_dim"]),
            hidden_dim=int(metadata["position_hidden_dim"]),
        )
    tensors = load_file(str(path / "selector.safetensors"))
    selector.load_state_dict({
        name.removeprefix("selector."): tensor
        for name, tensor in tensors.items() if name.startswith("selector.")
    }, strict=True)
    if positioner is not None:
        positioner.load_state_dict({
            name.removeprefix("positioner."): tensor
            for name, tensor in tensors.items() if name.startswith("positioner.")
        }, strict=True)
    return selector, positioner, metadata
