"""Stage-A supervised training for a query-aware LCLM BiEncoder selector.

The trainer consumes only (1) pre-Adapter ``LatentMemory`` Safetensors,
(2) short selector questions, (3) offline oracle latent labels, and optionally
(4) a full-latent contribution sidecar.  It never opens raw source context,
which keeps the training path aligned with the latent-only online retrieval
contract.  Evaluation benchmark labels must not be used here; prepare an
independently generated training split first.

Example (after building a *training* cache and its oracle sidecar)::

    python -m train.train_selector \
      --checkpoint artifacts/models/0.6b-4b-LCLM-8x \
      --cache-dir artifacts/data/selector_train/cache \
      --queries-jsonl artifacts/data/selector_train/queries.jsonl \
      --oracle-jsonl artifacts/data/selector_train/oracle_latents.jsonl \
      --contribution-jsonl artifacts/data/selector_train/contributions-zgrad.jsonl \
      --contribution-weight 1.0 --contribution-temperature 1.0 \
      --out artifacts/checkpoints/selector-stage-a
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import random
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F

from latent_context import (
    BiEncoderSelector,
    QuestionEncoder,
    document_id,
    from_pretrained,
    save_selector_checkpoint,
)
from latent_context.selector import LatentMemory, load_latent_memory, select_topk_in_source_order


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def multi_positive_ranking_loss(scores: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    """Put probability mass on every answer-containing latent.

    This is the stable multi-positive InfoNCE form: ``logsumexp(all) -
    logsumexp(positives)``.  It is paired with balanced BCE below because the
    listwise term alone can concentrate all mass on just one positive latent.
    """
    if scores.ndim != 1 or positive_mask.shape != scores.shape:
        raise ValueError("scores and positive_mask must both have shape [M]")
    if not bool(positive_mask.any()):
        raise ValueError("at least one positive latent is required")
    return torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[positive_mask], dim=0)


def balanced_bce_loss(scores: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    """Equal-weight positive/negative logistic loss for sparse oracle labels."""
    if not bool(positive_mask.any()) or bool(positive_mask.all()):
        return scores.new_zeros(())
    positive = -F.logsigmoid(scores[positive_mask]).mean()
    negative = -F.logsigmoid(-scores[~positive_mask]).mean()
    return 0.5 * (positive + negative)


def role_coverage_loss(
    scores: torch.Tensor,
    role_masks: dict[str, torch.Tensor],
    temperature: float = 1.0,
) -> torch.Tensor:
    """Encourage score mass for every offline evidence role.

    ``role_masks`` is an offline training label only (for example answer,
    key-answer, and support).  The loss is smooth and does not introduce an
    answer/oracle dependency into the online selector.
    """
    if scores.ndim != 1:
        raise ValueError("scores must be [M]")
    if not role_masks:
        return scores.new_zeros(())
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    log_probs = F.log_softmax(scores.float() / temperature, dim=0)
    losses = []
    for role, mask in role_masks.items():
        if mask.shape != scores.shape or mask.dtype != torch.bool:
            raise ValueError(f"role mask {role!r} must be bool [M]")
        if not bool(mask.any()):
            continue
        losses.append(-torch.logsumexp(log_probs[mask], dim=0))
    return torch.stack(losses).mean() if losses else scores.new_zeros(())


def group_coverage_loss(
    scores: torch.Tensor,
    group_masks: dict[str, torch.Tensor],
    temperature: float = 1.0,
    margin: float = 0.0,
    selection_k: int | None = None,
) -> torch.Tensor:
    """Encourage every evidence group to retain a high-scoring latent.

    ``role_coverage_loss`` treats a role as one large union.  That is useful
    for the historical answer/support labels, but it can still put all score
    mass on one member of a multi-key example.  This loss first assigns one
    smooth score to each *individual* group (``key_answer_0``,
    ``key_answer_1``, ...), normalising for group size, and applies a uniform
    cross entropy over groups.  When ``selection_k`` is supplied, a detached
    Top-K score threshold adds a direct surrogate for "at least one latent
    from every group is in Top-K".  All masks are offline labels and never
    enter the online answer path.

    Empty groups are ignored.  A missing/empty mapping returns a differentiable
    zero so callers can enable the term on mixed task families where some rows
    have no per-group annotations.
    """
    if scores.ndim != 1:
        raise ValueError("scores must be [M]")
    if not isinstance(group_masks, dict):
        raise ValueError("group_masks must be a mapping of names to bool [M] masks")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be finite")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if selection_k is not None and selection_k <= 0:
        raise ValueError("selection_k must be positive when supplied")

    valid_masks: list[torch.Tensor] = []
    for name, mask in group_masks.items():
        if not isinstance(mask, torch.Tensor) or mask.shape != scores.shape or mask.dtype != torch.bool:
            raise ValueError(f"group mask {name!r} must be bool [M]")
        if bool(mask.any()):
            valid_masks.append(mask)
    if not valid_masks:
        return scores.new_zeros(())

    scaled = scores.float() / temperature
    group_scores: list[torch.Tensor] = []
    for mask in valid_masks:
        # Mean-exp aggregation prevents a wider support window from winning
        # solely because it contains more latents.
        group_scores.append(torch.logsumexp(scaled[mask], dim=0) - math.log(int(mask.sum())))
    group_logits = torch.stack(group_scores)
    loss = -F.log_softmax(group_logits, dim=0).mean()

    # A detached threshold keeps this term stable while still supplying
    # gradients to every under-represented group.  It is intentionally a
    # surrogate, not an exact differentiable Top-K operation.
    if selection_k is not None and scores.numel() > 0:
        k = min(int(selection_k), int(scores.numel()))
        threshold = torch.topk(scores.detach().float(), k=k, dim=0).values[-1]
        group_peaks = group_logits * temperature
        loss = loss + F.softplus(
            (threshold + margin - group_peaks) / temperature
        ).mean()
    elif margin > 0 and len(group_scores) > 1:
        # Without a declared K, use a conservative pairwise margin between a
        # group peak and the strongest competing group.  This is only enabled
        # when explicitly requested, preserving the old objective by default.
        peaks = group_logits * temperature
        pairwise: list[torch.Tensor] = []
        for index, peak in enumerate(peaks):
            competitors = torch.cat((peaks[:index], peaks[index + 1:]))
            pairwise.append(F.relu(margin - peak + competitors.max()))
        loss = loss + torch.stack(pairwise).mean()
    return loss


def role_group_masks(
    role_masks: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    """Select numbered per-role groups from a role-mask mapping.

    Only names with the exact ``<prefix>_<integer>`` form are retained; the
    aggregate role (for example ``support``) is deliberately excluded.
    """
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("prefix must be a non-empty string")
    if not isinstance(role_masks, dict):
        raise ValueError("role_masks must be a mapping of names to masks")
    marker = prefix + "_"
    output: dict[str, torch.Tensor] = {}
    for name, mask in role_masks.items():
        if not isinstance(name, str) or not name.startswith(marker):
            continue
        suffix = name[len(marker):]
        # ``str.isdigit`` also accepts non-ASCII numerals (and a few unicode
        # numeric characters), which makes sidecar role names ambiguous.  Role
        # keys are deliberately restricted to the exact ASCII form expected
        # from the annotation tools: ``<prefix>_<integer>``.
        if suffix and suffix.isascii() and suffix.isdecimal():
            output[name] = mask
    return output


# Descriptive alias used by experiment scripts and downstream diagnostics.
per_group_coverage_loss = group_coverage_loss


def position_balance_loss(scores: torch.Tensor, num_buckets: int = 8) -> torch.Tensor:
    """Penalize a selector distribution that collapses into one source bucket."""
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a non-empty [M] tensor")
    if num_buckets <= 0:
        raise ValueError("num_buckets must be positive")
    buckets = torch.tensor_split(torch.arange(scores.numel(), device=scores.device), min(num_buckets, scores.numel()))
    probs = F.softmax(scores.float(), dim=0)
    masses = torch.stack([probs[indexes].sum() for indexes in buckets])
    target = torch.full_like(masses, 1.0 / masses.numel())
    return F.kl_div(masses.clamp_min(torch.finfo(masses.dtype).tiny).log(), target, reduction="batchmean")


def utility_candidate_rank_loss(
    scores: torch.Tensor,
    candidates: list[dict[str, Any]],
    margin: float = 0.1,
) -> torch.Tensor:
    """Distill answer-NLL gains between explicit candidate latent sets.

    Each candidate has ``latent_indices`` (or ``candidate_latent_indices``) and
    a numeric ``gain``/``candidate_gain``.  The finite candidate family is an
    offline diagnostic target; no candidate NLL is read by answer-time code.
    """
    if scores.ndim != 1 or margin < 0:
        raise ValueError("scores must be [M] and margin must be non-negative")
    parsed: list[tuple[float, torch.Tensor]] = []
    for candidate in candidates:
        raw_indices = candidate.get("latent_indices", candidate.get("candidate_latent_indices"))
        raw_gain = candidate.get("gain", candidate.get("candidate_gain"))
        if not isinstance(raw_indices, list) or raw_gain is None:
            raise ValueError("utility candidates require latent_indices and gain")
        valid = sorted({int(index) for index in raw_indices if 0 <= int(index) < scores.numel()})
        if not valid:
            continue
        mask = torch.zeros(scores.numel(), dtype=torch.bool, device=scores.device)
        mask[torch.tensor(valid, dtype=torch.long, device=scores.device)] = True
        parsed.append((float(raw_gain), mask))
    if len(parsed) < 2:
        return scores.new_zeros(())
    losses = []
    for left_index, (left_gain, left_mask) in enumerate(parsed):
        left_score = torch.logsumexp(scores[left_mask], dim=0)
        for right_gain, right_mask in parsed[left_index + 1:]:
            right_score = torch.logsumexp(scores[right_mask], dim=0)
            if left_gain == right_gain:
                continue
            if left_gain > right_gain:
                losses.append(F.relu(torch.as_tensor(margin, device=scores.device) - left_score + right_score))
            else:
                losses.append(F.relu(torch.as_tensor(margin, device=scores.device) - right_score + left_score))
    return torch.stack(losses).mean() if losses else scores.new_zeros(())


CONTRIBUTION_NORMALIZATIONS = ("zgrad", "minmax", "softmax")


def contribution_target_distribution(
    contributions: torch.Tensor,
    temperature: float = 1.0,
    normalization: str = "zgrad",
) -> torch.Tensor:
    """Convert offline contribution magnitudes into a detached soft target.

    ``contributions`` may be ``[M]`` or ``[B, M]``.  ``zgrad`` keeps the raw
    non-negative first-order magnitudes, ``minmax`` rescales each row to
    ``[0, 1]``, and ``softmax`` is an explicit alias for applying softmax to
    the raw magnitudes.  Temperature is applied to both teacher and student
    distributions by :func:`contribution_distillation_loss`.
    """
    if contributions.ndim not in (1, 2) or contributions.shape[-1] == 0:
        raise ValueError("contributions must have shape [M] or [B, M]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if normalization not in CONTRIBUTION_NORMALIZATIONS:
        raise ValueError(
            f"normalization must be one of {CONTRIBUTION_NORMALIZATIONS}, got {normalization!r}"
        )
    values = contributions.float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("contributions must be finite")
    if bool((values < 0).any()):
        raise ValueError("contributions must be non-negative")
    if normalization == "minmax":
        minimum = values.amin(dim=-1, keepdim=True)
        spread = values.amax(dim=-1, keepdim=True) - minimum
        values = torch.where(spread > 0, (values - minimum) / spread, torch.zeros_like(values))
    # The returned tensor is a label.  Detaching also prevents accidental
    # coupling if a caller passes a tensor that still belongs to another graph.
    return F.softmax(values / temperature, dim=-1).detach()


def contribution_distillation_loss(
    scores: torch.Tensor,
    contributions: torch.Tensor,
    temperature: float = 1.0,
    normalization: str = "zgrad",
) -> torch.Tensor:
    """Teacher||student KL for one selector score vector or a batch.

    Oracle span rank/BCE losses remain the hard target.  This auxiliary term
    gives probability mass to every latent in proportion to its offline
    full-latent answer-loss contribution.
    """
    if scores.shape != contributions.shape or scores.ndim not in (1, 2):
        raise ValueError("scores and contributions must have the same shape [M] or [B, M]")
    target = contribution_target_distribution(contributions, temperature, normalization)
    student_log_probs = F.log_softmax(scores.float() / temperature, dim=-1)
    kl = target * (target.clamp_min(torch.finfo(target.dtype).tiny).log() - student_log_probs)
    return kl.sum(dim=-1).mean()


class _MemoryLRU:
    """CPU-resident cache LRU; GPU receives only the current training item."""

    def __init__(self, cache_dir: str | Path, capacity: int = 32) -> None:
        self.root = Path(cache_dir)
        self.capacity = capacity
        self._items: OrderedDict[str, LatentMemory] = OrderedDict()

    def get(self, record_id: str, device: str) -> LatentMemory:
        if record_id in self._items:
            memory = self._items.pop(record_id)
            self._items[record_id] = memory
        else:
            memory = load_latent_memory(self.root / record_id)
            self._items[record_id] = memory
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        return memory.to(device)


def _load_contribution_targets(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    targets: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        record_id = str(row["id"])
        if record_id in targets:
            raise ValueError(f"duplicate contribution target for {record_id}")
        values = row.get("latent_contributions")
        if not isinstance(values, list) or not values:
            raise ValueError(f"contribution target for {record_id} is empty or malformed")
        try:
            values = [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"contribution target for {record_id} is not numeric") from exc
        targets[record_id] = {
            "values": values,
            "target_definition": row.get("target_definition"),
            "cache_fingerprint": row.get("cache_fingerprint"),
            "schema_version": row.get("contribution_schema_version"),
            "position_policy": row.get("position_policy"),
            "candidate_sets": row.get("candidate_sets", row.get("utility_candidates", [])),
            "document_id": row.get("document_id"),
        }
    if not targets:
        raise ValueError("contribution sidecar contains no rows")
    return targets


def _load_utility_targets(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load optional finite candidate-family utility supervision."""
    if path is None:
        return {}
    targets: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        record_id = str(row["id"])
        if record_id in targets:
            raise ValueError(f"duplicate utility target for {record_id}")
        candidates = row.get("candidate_answer_nll", row.get("utility_candidates", []))
        if not isinstance(candidates, list):
            raise ValueError(f"utility target for {record_id} has malformed candidates")
        full_nll = row.get("answer_nll_full", row.get("full_nll"))
        normalized = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError(f"utility target for {record_id} contains a malformed candidate")
            names = candidate.get("names", candidate.get("candidate_role", []))
            if isinstance(names, str):
                names = [names]
            indices = candidate.get("latent_indices", candidate.get("candidate_latent_indices"))
            nll = candidate.get("answer_nll", candidate.get("selected_nll"))
            if not isinstance(indices, list) or nll is None:
                continue
            gain = (float(full_nll) - float(nll)) if full_nll is not None else -float(nll)
            normalized.append({
                "latent_indices": indices,
                "gain": gain,
                "candidate_role": list(names) if isinstance(names, list) else [],
                "answer_nll": float(nll),
            })
        targets[record_id] = {
            "candidates": normalized,
            "cache_fingerprint": row.get("cache_fingerprint"),
            "position_policy": row.get("position_policy"),
            "schema_version": row.get("utility_oracle_schema_version"),
            "document_id": row.get("document_id"),
        }
    return targets


def _build_examples(
    queries_path: str | Path,
    labels_path: str | Path,
    contribution_path: str | Path | None = None,
    max_examples: int = 0,
    utility_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for label_row in _read_jsonl(labels_path):
        label_id = str(label_row["id"])
        if label_id in labels:
            raise ValueError(f"duplicate oracle label for {label_id}")
        labels[label_id] = label_row
    contributions = _load_contribution_targets(contribution_path)
    utilities = _load_utility_targets(utility_path)
    examples = []
    for row in _read_jsonl(queries_path):
        if "context" in row:
            raise ValueError("queries-jsonl must not contain raw context")
        record_id = str(row["id"])
        question = row.get("selector_query", row.get("question"))
        if not isinstance(question, str) or not question:
            raise ValueError(f"missing selector question for {record_id}")
        label_row = labels.get(record_id, {})
        query_document = document_id(row)
        label_document = label_row.get("document_id")
        if label_document is not None and str(label_document) != query_document:
            raise ValueError(f"oracle label {record_id} points to the wrong document")
        # Keep the historical hard target as the union of all answer spans.
        # The separate ``answer`` role is available to the optional role loss,
        # while changing the default rank target would silently drop additional
        # answers from existing training runs.
        positives = sorted({int(index) for index in label_row.get("oracle_latent_indices", [])})
        if positives:
            example = {
                "id": record_id,
                "document_id": query_document,
                "question": question,
                "positives": positives,
                "split": row.get("split"),
                "task": row.get("task"),
                "roles": label_row.get("role_latent_indices", label_row.get("roles", {})),
                "oracle_cache_fingerprint": label_row.get("cache_fingerprint"),
                "oracle_cache_schema_version": label_row.get("cache_schema_version"),
            }
            if contribution_path is not None:
                target = contributions.get(record_id)
                if target is None:
                    raise ValueError(f"missing contribution target for {record_id}")
                target_document = target.get("document_id")
                if target_document is not None and str(target_document) != query_document:
                    raise ValueError(f"contribution target {record_id} points to the wrong document")
                example["contribution"] = target
            if utility_path is not None:
                target = utilities.get(record_id)
                if target is None:
                    raise ValueError(f"missing utility target for {record_id}")
                target_document = target.get("document_id")
                if target_document is not None and str(target_document) != query_document:
                    raise ValueError(f"utility target {record_id} points to the wrong document")
                example["utility"] = target
            examples.append(example)
            if max_examples and len(examples) >= max_examples:
                break
    if not examples:
        raise ValueError("no examples with oracle positives; check the sidecar and cache tokenization")
    return examples


def split_examples_by_document(
    examples: list[dict[str, Any]],
    validation_fraction: float,
    split_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Split whole documents, or honor a fully specified train/validation split."""
    documents: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        documents.setdefault(str(example["document_id"]), []).append(example)

    explicit = [example.get("split") for example in examples]
    if any(value is not None for value in explicit):
        if any(value not in {"train", "validation"} for value in explicit):
            raise ValueError("explicit split must be present on every query and be train/validation")
        document_splits = {}
        for memory_id, items in documents.items():
            values = {item["split"] for item in items}
            if len(values) != 1:
                raise ValueError(f"document {memory_id} appears in multiple explicit splits")
            document_splits[memory_id] = next(iter(values))
        split_mode = "explicit_document_split"
    else:
        document_ids = sorted(documents)
        random.Random(split_seed).shuffle(document_ids)
        validation_count = int(len(document_ids) * validation_fraction)
        if validation_fraction > 0 and validation_count == 0 and len(document_ids) > 1:
            validation_count = 1
        validation_ids = set(document_ids[:validation_count])
        document_splits = {
            memory_id: ("validation" if memory_id in validation_ids else "train")
            for memory_id in document_ids
        }
        split_mode = "seeded_group_split"

    train_examples = [
        example for example in examples if document_splits[str(example["document_id"])] == "train"
    ]
    validation_examples = [
        example for example in examples
        if document_splits[str(example["document_id"])] == "validation"
    ]
    if not train_examples:
        raise ValueError("document split left no training examples")
    train_documents = sorted({str(example["document_id"]) for example in train_examples})
    validation_documents = sorted({str(example["document_id"]) for example in validation_examples})
    overlap = sorted(set(train_documents) & set(validation_documents))
    if overlap:
        raise AssertionError(f"document leakage across train/validation: {overlap[:3]}")
    manifest = {
        "schema_version": 1,
        "split_mode": split_mode,
        "split_seed": split_seed,
        "validation_fraction_requested": validation_fraction,
        "num_train_queries": len(train_examples),
        "num_validation_queries": len(validation_examples),
        "num_train_documents": len(train_documents),
        "num_validation_documents": len(validation_documents),
        "train_document_ids": train_documents,
        "validation_document_ids": validation_documents,
        "document_overlap": overlap,
    }
    return train_examples, validation_examples, manifest


def _positive_mask(memory: LatentMemory, indices: list[int], device: str) -> torch.Tensor:
    mask = torch.zeros(memory.num_latents, dtype=torch.bool, device=device)
    valid = [index for index in indices if 0 <= index < memory.num_latents]
    if not valid:
        raise ValueError("oracle labels have no valid indices for this cache")
    mask[torch.tensor(valid, dtype=torch.long, device=device)] = True
    return mask


def _role_masks(
    memory: LatentMemory,
    roles: Any,
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    if not isinstance(roles, dict):
        return {}
    output: dict[str, torch.Tensor] = {}
    for role, raw_indices in roles.items():
        if not isinstance(raw_indices, list):
            continue
        mask = torch.zeros(memory.num_latents, dtype=torch.bool, device=device)
        valid = sorted({int(index) for index in raw_indices if 0 <= int(index) < memory.num_latents})
        if valid:
            mask[torch.tensor(valid, dtype=torch.long, device=device)] = True
            output[str(role)] = mask
    return output


def _contribution_tensor(
    memory: LatentMemory,
    target: dict[str, Any],
    device: str | torch.device,
    expected_fingerprint: str,
) -> torch.Tensor:
    values = target["values"]
    if len(values) != memory.num_latents:
        raise ValueError(
            f"contribution target length {len(values)} != cache {memory.num_latents}"
        )
    target_fingerprint = target.get("cache_fingerprint")
    if target_fingerprint and target_fingerprint != expected_fingerprint:
        raise ValueError("contribution target was built by an incompatible Encoder")
    schema_version = target.get("schema_version")
    if schema_version is not None and int(schema_version) not in {1, 2}:
        raise ValueError(f"unsupported contribution sidecar schema: {schema_version}")
    if target.get("position_policy") not in (None, "full_layout"):
        raise ValueError("contribution target must use full_layout or omit position_policy")
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    if not bool(torch.isfinite(tensor).all()) or bool((tensor < 0).any()):
        raise ValueError("contribution target must contain finite non-negative values")
    return tensor


@torch.no_grad()
def evaluate_recall(
    examples: list[dict[str, Any]],
    memory_lru: _MemoryLRU,
    question_encoder: QuestionEncoder,
    selector: BiEncoderSelector,
    tokenizer,
    device: str,
    selection_ratio: float,
) -> dict[str, float]:
    selector.eval()
    recalls: list[float] = []
    role_values: dict[str, list[float]] = {}
    for example in examples:
        memory = memory_lru.get(example["document_id"], device)
        encoded = tokenizer([example["question"]], padding=True, return_tensors="pt", add_special_tokens=False)
        question = question_encoder(
            encoded["input_ids"].to(device), encoded["attention_mask"].to(device)
        )
        k = max(1, math.ceil(selection_ratio * memory.num_latents))
        selection = select_topk_in_source_order(selector.score(question, memory), k)
        selected = set(selection.indices[0].tolist())
        positives = {index for index in example["positives"] if 0 <= index < memory.num_latents}
        recalls.append(len(selected & positives) / len(positives))
        roles = example.get("roles")
        if isinstance(roles, dict):
            for role, raw_indices in roles.items():
                if not isinstance(raw_indices, list):
                    continue
                role_indices = {
                    int(index) for index in raw_indices
                    if 0 <= int(index) < memory.num_latents
                }
                if role_indices:
                    role_values.setdefault(str(role), []).append(
                        len(selected & role_indices) / len(role_indices)
                    )
    selector.train()
    metrics: dict[str, float] = {
        "oracle_latent_recall": sum(recalls) / len(recalls),
        "examples": float(len(recalls)),
    }
    for role, values in sorted(role_values.items()):
        metrics[f"{role}_recall"] = sum(values) / len(values)
        metrics[f"{role}_recall_examples"] = float(len(values))
    return metrics


def _validate_utility_target(
    target: dict[str, Any],
    expected_fingerprint: str,
) -> None:
    target_fingerprint = target.get("cache_fingerprint")
    if target_fingerprint and target_fingerprint != expected_fingerprint:
        raise ValueError("utility target was built by an incompatible Encoder")
    schema_version = target.get("schema_version")
    if schema_version is not None and int(schema_version) not in {1, 2}:
        raise ValueError(f"unsupported utility sidecar schema: {schema_version}")
    position_policy = target.get("position_policy")
    if position_policy not in (None, "full_layout"):
        raise ValueError("utility target must use full_layout or omit position_policy")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Stage-A query-aware latent BiEncoder")
    parser.add_argument("--checkpoint", required=True, help="BF16/FP16 LCLM checkpoint, never GGUF")
    parser.add_argument("--cache-dir", required=True, help="pre-Adapter LatentMemory cache directory")
    parser.add_argument("--queries-jsonl", required=True, help="latent-only selector questions")
    parser.add_argument("--oracle-jsonl", required=True, help="offline answer-span latent sidecar")
    parser.add_argument("--contribution-jsonl", default=None,
                        help="offline full-latent contribution sidecar (optional)")
    parser.add_argument("--utility-jsonl", default=None,
                        help="offline finite candidate-family utility sidecar (optional)")
    parser.add_argument("--out", required=True, help="selector checkpoint directory")
    parser.add_argument("--max-examples", type=int, default=0,
                        help="0 uses all labeled examples; useful for contribution smoke runs")
    parser.add_argument("--retrieval-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--contribution-weight", type=float, default=None,
                        help="weight for contribution KL; defaults to 1 when sidecar is provided")
    parser.add_argument("--contribution-temperature", type=float, default=1.0)
    parser.add_argument("--contribution-normalization", choices=CONTRIBUTION_NORMALIZATIONS,
                        default="zgrad")
    parser.add_argument("--utility-weight", type=float, default=0.0,
                        help="weight for explicit candidate utility ranking loss")
    parser.add_argument("--role-coverage-weight", type=float, default=0.0,
                        help="weight for answer/key/support role coverage loss")
    parser.add_argument("--key-answer-group-weight", type=float, default=0.0,
                        help="weight for per-key key-answer group coverage loss")
    parser.add_argument("--support-group-weight", type=float, default=0.0,
                        help="weight for per-key support group coverage loss")
    parser.add_argument("--group-coverage-temperature", type=float, default=1.0,
                        help="temperature for per-group coverage losses")
    parser.add_argument("--group-max-margin", type=float, default=0.0,
                        help="optional margin used by per-group coverage losses")
    parser.add_argument("--group-selection-ratio", type=float, default=None,
                        help="selection ratio used by per-group Top-K surrogate; defaults to eval ratio")
    parser.add_argument("--position-balance-weight", type=float, default=0.0,
                        help="weight for source-position balance regularization")
    parser.add_argument("--role-temperature", type=float, default=1.0)
    parser.add_argument("--position-balance-buckets", type=int, default=8)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--eval-selection-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=4231)
    parser.add_argument("--split-seed", type=int, default=6201,
                        help="fixed document split seed, independent of optimization seed")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()

    if args.epochs <= 0 or args.grad_accumulation <= 0 or args.retrieval_dim <= 0:
        raise ValueError("epochs, grad accumulation, and retrieval dimension must be positive")
    if args.max_examples < 0:
        raise ValueError("max-examples must be non-negative")
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("validation fraction must be in [0, 1)")
    if not 0.0 < args.eval_selection_ratio <= 1.0:
        raise ValueError("eval selection ratio must be in (0, 1]")
    if args.rank_weight < 0 or args.bce_weight < 0:
        raise ValueError("rank and BCE weights must be non-negative")
    contribution_weight = (
        (1.0 if args.contribution_jsonl is not None else 0.0)
        if args.contribution_weight is None else args.contribution_weight
    )
    if contribution_weight < 0:
        raise ValueError("contribution weight must be non-negative")
    if args.contribution_temperature <= 0:
        raise ValueError("contribution temperature must be positive")
    if contribution_weight > 0 and args.contribution_jsonl is None:
        raise ValueError("--contribution-jsonl is required when contribution weight is non-zero")
    if (
        args.utility_weight < 0
        or args.role_coverage_weight < 0
        or args.key_answer_group_weight < 0
        or args.support_group_weight < 0
        or args.position_balance_weight < 0
    ):
        raise ValueError("auxiliary loss weights must be non-negative")
    if args.utility_weight > 0 and args.utility_jsonl is None:
        raise ValueError("--utility-jsonl is required when --utility-weight is non-zero")
    if (
        args.role_temperature <= 0
        or args.group_coverage_temperature <= 0
        or args.group_max_margin < 0
        or args.position_balance_buckets <= 0
    ):
        raise ValueError("role/group temperatures and margin must be valid, and position buckets positive")
    if args.group_selection_ratio is not None and not 0.0 < args.group_selection_ratio <= 1.0:
        raise ValueError("group selection ratio must be in (0, 1]")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    examples = _build_examples(
        args.queries_jsonl,
        args.oracle_jsonl,
        args.contribution_jsonl,
        max_examples=args.max_examples,
        utility_path=args.utility_jsonl,
    )
    train_examples, val_examples, split_manifest = split_examples_by_document(
        examples, args.validation_fraction, args.split_seed
    )

    model, _, _ = from_pretrained(args.checkpoint, device=args.device, dtype=args.dtype)
    model.eval()
    hidden_size = model.encoder.embed_model.config.hidden_size
    question_encoder = QuestionEncoder(model.encoder.embed_model, hidden_size, freeze_backbone=True).to(args.device)
    question_encoder.eval()
    selector = BiEncoderSelector(hidden_size, retrieval_dim=args.retrieval_dim).to(args.device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=args.learning_rate, weight_decay=0.01)
    memory_lru = _MemoryLRU(args.cache_dir)
    tokenizer = model.encoder.embed_tokenizer
    fingerprint = model.memory_fingerprint()

    history = []
    global_step = 0
    start = perf_counter()
    for epoch in range(1, args.epochs + 1):
        random.Random(args.seed + epoch).shuffle(train_examples)
        selector.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = total_rank = total_bce = total_distill = 0.0
        total_utility = total_role = total_key_group = total_support_group = total_balance = 0.0
        for item_index, example in enumerate(train_examples, start=1):
            memory = memory_lru.get(example["document_id"], args.device)
            if memory.fingerprint != fingerprint:
                raise ValueError(
                    f"cache {example['document_id']} was built by an incompatible Encoder"
                )
            if example.get("oracle_cache_fingerprint") not in (None, memory.fingerprint):
                raise ValueError("oracle labels were built by an incompatible Encoder")
            oracle_schema = example.get("oracle_cache_schema_version")
            if oracle_schema is not None and int(oracle_schema) != memory.schema_version:
                raise ValueError("oracle labels use an incompatible cache schema")
            encoded = tokenizer([example["question"]], padding=True, return_tensors="pt", add_special_tokens=False)
            question = question_encoder(
                encoded["input_ids"].to(args.device), encoded["attention_mask"].to(args.device)
            )
            scores = selector.score(question, memory)[0]
            positives = _positive_mask(memory, example["positives"], args.device)
            rank_loss = multi_positive_ranking_loss(scores, positives)
            bce_loss = balanced_bce_loss(scores, positives)
            distill_loss = scores.new_zeros(())
            if contribution_weight > 0:
                contribution = _contribution_tensor(
                    memory,
                    example["contribution"],
                    args.device,
                    fingerprint,
                )
                distill_loss = contribution_distillation_loss(
                    scores,
                    contribution,
                    temperature=args.contribution_temperature,
                    normalization=args.contribution_normalization,
                )
            role_masks = _role_masks(memory, example.get("roles"), args.device)
            role_loss = role_coverage_loss(
                scores,
                role_masks,
                temperature=args.role_temperature,
            ) if args.role_coverage_weight > 0 else scores.new_zeros(())
            group_selection_ratio = (
                args.group_selection_ratio
                if args.group_selection_ratio is not None
                else args.eval_selection_ratio
            )
            group_selection_k = max(1, math.ceil(group_selection_ratio * memory.num_latents))
            key_answer_group_loss = group_coverage_loss(
                scores,
                role_group_masks(role_masks, "key_answer"),
                temperature=args.group_coverage_temperature,
                margin=args.group_max_margin,
                selection_k=group_selection_k,
            ) if args.key_answer_group_weight > 0 else scores.new_zeros(())
            support_group_loss = group_coverage_loss(
                scores,
                role_group_masks(role_masks, "support"),
                temperature=args.group_coverage_temperature,
                margin=args.group_max_margin,
                selection_k=group_selection_k,
            ) if args.support_group_weight > 0 else scores.new_zeros(())
            utility_candidates = []
            utility_target = example.get("utility")
            if utility_target:
                _validate_utility_target(utility_target, fingerprint)
                utility_candidates.extend(utility_target.get("candidates", []))
            if example.get("contribution", {}).get("candidate_sets"):
                utility_candidates.extend(example["contribution"]["candidate_sets"])
            utility_loss = utility_candidate_rank_loss(scores, utility_candidates) \
                if args.utility_weight > 0 else scores.new_zeros(())
            balance_loss = position_balance_loss(
                scores, args.position_balance_buckets
            ) if args.position_balance_weight > 0 else scores.new_zeros(())
            loss = (
                args.rank_weight * rank_loss
                + args.bce_weight * bce_loss
                + contribution_weight * distill_loss
                + args.utility_weight * utility_loss
                + args.role_coverage_weight * role_loss
                + args.key_answer_group_weight * key_answer_group_loss
                + args.support_group_weight * support_group_loss
                + args.position_balance_weight * balance_loss
            )
            (loss / args.grad_accumulation).backward()
            if item_index % args.grad_accumulation == 0 or item_index == len(train_examples):
                torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            total_loss += float(loss.detach())
            total_rank += float(rank_loss.detach())
            total_bce += float(bce_loss.detach())
            total_distill += float(distill_loss.detach())
            total_utility += float(utility_loss.detach())
            total_role += float(role_loss.detach())
            total_key_group += float(key_answer_group_loss.detach())
            total_support_group += float(support_group_loss.detach())
            total_balance += float(balance_loss.detach())
            if global_step and global_step % args.log_interval == 0 and item_index % args.grad_accumulation == 0:
                print(json.dumps({"epoch": epoch, "step": global_step, "loss": total_loss / item_index}), flush=True)

        train_metrics = {
            "loss": total_loss / len(train_examples),
            "rank_loss": total_rank / len(train_examples),
            "balanced_bce": total_bce / len(train_examples),
            "contribution_kl": total_distill / len(train_examples),
            "utility_rank_loss": total_utility / len(train_examples),
            "role_coverage_loss": total_role / len(train_examples),
            "key_answer_group_coverage_loss": total_key_group / len(train_examples),
            "support_group_coverage_loss": total_support_group / len(train_examples),
            "position_balance_loss": total_balance / len(train_examples),
        }
        val_metrics = (
            evaluate_recall(val_examples, memory_lru, question_encoder, selector, tokenizer,
                            args.device, args.eval_selection_ratio)
            if val_examples else {}
        )
        row = {"epoch": epoch, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(json.dumps(row), flush=True)

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    split_payload = json.dumps(split_manifest, indent=2) + "\n"
    (output / "data_split.json").write_text(split_payload)
    split_sha256 = hashlib.sha256(split_payload.encode()).hexdigest()
    contribution_rows = [item.get("contribution", {}) for item in examples if item.get("contribution")]
    contribution_definitions = sorted({
        row.get("target_definition") for row in contribution_rows if row.get("target_definition")
    })
    contribution_position_policies = sorted({
        row.get("position_policy") for row in contribution_rows if row.get("position_policy")
    })
    sidecar_sha256 = None
    if args.contribution_jsonl:
        digest = hashlib.sha256()
        with Path(args.contribution_jsonl).open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        sidecar_sha256 = digest.hexdigest()
    utility_sidecar_sha256 = None
    if args.utility_jsonl:
        digest = hashlib.sha256()
        with Path(args.utility_jsonl).open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        utility_sidecar_sha256 = digest.hexdigest()
    training_stage = (
        "A+combination_utility+group_coverage"
        if (
            args.utility_weight > 0
            or args.role_coverage_weight > 0
            or args.key_answer_group_weight > 0
            or args.support_group_weight > 0
        )
        else "A+contribution_distillation" if contribution_weight > 0 else "A"
    )
    checkpoint_metadata = {
        "training_stage": training_stage,
        "contribution_distillation": contribution_weight > 0,
        "contribution_weight": contribution_weight,
        "contribution_temperature": args.contribution_temperature,
        "contribution_normalization": args.contribution_normalization,
        "contribution_target_definition": contribution_definitions[0]
        if len(contribution_definitions) == 1 else contribution_definitions,
        "contribution_position_policy": contribution_position_policies[0]
        if len(contribution_position_policies) == 1 else contribution_position_policies,
        "contribution_sidecar_sha256": sidecar_sha256,
        "utility_sidecar_sha256": utility_sidecar_sha256,
        "document_split_sha256": split_sha256,
        "split_seed": args.split_seed,
        "utility_weight": args.utility_weight,
        "role_coverage_weight": args.role_coverage_weight,
        "key_answer_group_weight": args.key_answer_group_weight,
        "support_group_weight": args.support_group_weight,
        "group_coverage_temperature": args.group_coverage_temperature,
        "group_max_margin": args.group_max_margin,
        "group_selection_ratio": args.group_selection_ratio,
        "position_balance_weight": args.position_balance_weight,
        "role_temperature": args.role_temperature,
        "position_balance_buckets": args.position_balance_buckets,
    }
    save_selector_checkpoint(output, selector, None, fingerprint, checkpoint_metadata)
    (output / "training_metrics.json").write_text(json.dumps({
        "stage": training_stage,
        "seed": args.seed, "num_train_examples": len(train_examples),
        "num_validation_examples": len(val_examples), "elapsed_seconds": perf_counter() - start,
        "document_split": split_manifest,
        "history": history,
        "input_contract": "cached latent vectors + short questions + offline oracle labels only",
        "contribution_distillation": checkpoint_metadata,
    }, indent=2) + "\n")
    print(json.dumps({"saved": str(output), "final": history[-1]}), flush=True)


if __name__ == "__main__":
    main()
