"""Latent-only query-aware inference for LCLM.

The ``cache`` command reads source text exactly once to construct a
``LatentMemory``.  The ``answer`` command only opens that cache and questions;
it has no context-text input, enforcing the experimental constraint that the
selector cannot reread the original document.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Iterable

import torch

from latent_context import (
    DEFAULT_POSITION_POLICY, POSITION_POLICIES, CosineSelector, QuestionEncoder,
    answer_nll_metrics, build_decoder_position_ids, document_id, from_pretrained,
    memory_structure_groups, coverage_aware_greedy, coverage_for_selection,
    source_span_summary,
    load_selector_checkpoint,
    select_dynamic_budget, select_random_indices, select_uniform_indices,
)
from latent_context.selector import (
    content_token_slots, extract_entity_slots, load_latent_memory, save_latent_memory,
    reciprocal_rank_fusion, select_slots_union, split_question_sentences,
)

SLOT_POLICIES = ("none", "entity", "sentence", "content_tokens")
SLOT_COMBINES = ("union", "rank_fusion", "max")


def _sync_if_cuda(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_oracle_labels(path: str | Path | None) -> dict[str, list[int]]:
    if path is None:
        return {}
    labels = {}
    for row in _read_jsonl(path):
        record_id = str(row["id"])
        if record_id in labels:
            raise ValueError(f"duplicate oracle label for {record_id}")
        if row.get("oracle_schema_version") not in (None, 1):
            raise ValueError(f"unsupported oracle schema for {record_id}")
        labels[record_id] = [int(index) for index in row.get("oracle_latent_indices", [])]
    return labels


def _load_oracle_roles(path: str | Path | None) -> dict[str, dict[str, list[int]]]:
    """Load optional offline role labels for diagnostics only."""
    if path is None:
        return {}
    roles: dict[str, dict[str, list[int]]] = {}
    for row in _read_jsonl(path):
        record_id = str(row["id"])
        if record_id in roles:
            raise ValueError(f"duplicate oracle role label for {record_id}")
        if row.get("oracle_schema_version") not in (None, 1):
            raise ValueError(f"unsupported oracle schema for {record_id}")
        role_rows = row.get("role_latent_indices", {})
        normalized = {
            str(role): [int(index) for index in indices]
            for role, indices in role_rows.items()
            if isinstance(indices, list)
        } if isinstance(role_rows, dict) else {}
        if "answer" not in normalized:
            normalized["answer"] = [
                int(index) for index in row.get(
                    "answer_latent_indices", row.get("oracle_latent_indices", [])
                )
            ]
        if "all_answer" not in normalized:
            normalized["all_answer"] = [
                int(index) for index in row.get("all_answer_latent_indices", normalized["answer"])
            ]
        roles[record_id] = normalized
    return roles


def _oracle_role_indices(
    roles: dict[str, list[int]],
    default_indices: list[int],
    role_name: str | None,
) -> list[int]:
    """Resolve an offline role-oracle target without entering the online path.

    ``role_name`` may name one role (for example ``key_answer``) or a
    comma-separated union (``key_answer,support``).  The latter is useful for
    the key-answer-support diagnostic and remains auditable because every
    component must be present in the role sidecar.  An unavailable role is an
    error rather than an empty target, preventing a mislabeled oracle run from
    looking like a valid zero-recall result.
    """
    if not role_name:
        return list(default_indices)
    names = [name.strip() for name in role_name.split(",") if name.strip()]
    if not names:
        raise ValueError("--oracle-role must contain at least one role name")
    missing = [name for name in names if name not in roles]
    if missing:
        available = ", ".join(sorted(roles)) or "<none>"
        raise ValueError(
            f"oracle role(s) {', '.join(missing)} unavailable; available roles: {available}"
        )
    return sorted({int(index) for name in names for index in roles[name]})


def _load_oracle_fingerprints(path: str | Path | None) -> dict[str, tuple[str | None, int | None]]:
    if path is None:
        return {}
    fingerprints: dict[str, tuple[str | None, int | None]] = {}
    for row in _read_jsonl(path):
        record_id = str(row["id"])
        if record_id in fingerprints:
            raise ValueError(f"duplicate oracle provenance row for {record_id}")
        schema = row.get("oracle_schema_version")
        if schema not in (None, 1):
            raise ValueError(f"unsupported oracle schema for {record_id}")
        fingerprints[record_id] = (
            row.get("cache_fingerprint"),
            int(row["cache_schema_version"]) if row.get("cache_schema_version") is not None else None,
        )
    return fingerprints


def _load_utility_oracle(path: str | Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    rows = {}
    for row in _read_jsonl(path):
        record_id = str(row["id"])
        if record_id in rows:
            raise ValueError(f"duplicate utility-oracle row for {record_id}")
        indices = row.get("utility_oracle_latent_indices")
        if not isinstance(indices, list):
            raise ValueError(f"utility-oracle row {record_id} has no latent indices")
        if row.get("utility_oracle_schema_version") not in (None, 1, 2):
            raise ValueError(f"unsupported utility-oracle schema for {record_id}")
        rows[record_id] = row
    return rows


def _requested_k(memory_length: int, k_decoder: int | None, selection_ratio: float | None) -> int:
    if (k_decoder is None) == (selection_ratio is None):
        raise ValueError("provide exactly one of --k-decoder or --selection-ratio")
    if selection_ratio is not None:
        if not 0.0 < selection_ratio <= 1.0:
            raise ValueError("--selection-ratio must be in (0, 1]")
        return max(1, math.ceil(selection_ratio * memory_length))
    if k_decoder is None or k_decoder <= 0:
        raise ValueError("--k-decoder must be positive")
    return min(k_decoder, memory_length)


def _oracle_fill_indices(
    oracle_indices: list[int], memory_length: int, k: int, device: str | torch.device,
) -> torch.Tensor:
    """Oracle positives plus deterministic uniform filler under a fixed budget."""
    actual_k = min(k, memory_length)
    positives = sorted({index for index in oracle_indices if 0 <= index < memory_length})
    if len(positives) >= actual_k:
        return torch.tensor(positives[:actual_k], dtype=torch.long, device=device)
    chosen = set(positives)
    # Prefer broad source coverage for the non-oracle budget; this avoids
    # granting Oracle a hidden contiguous-context advantage.
    for index in select_uniform_indices(memory_length, min(memory_length, actual_k * 4)).tolist():
        chosen.add(index)
        if len(chosen) >= actual_k:
            break
    if len(chosen) < actual_k:
        for index in range(memory_length):
            chosen.add(index)
            if len(chosen) >= actual_k:
                break
    return torch.tensor(sorted(chosen), dtype=torch.long, device=device)


def _estimate_kv_cache_bytes(config, sequence_length: int, dtype: torch.dtype) -> int:
    """Return the decoder-only K/V cache estimate for one batch element.

    This is intentionally an architectural estimate rather than a CUDA memory
    counter: allocator fragmentation and model activations are reported
    separately through ``peak_memory_bytes``.  Qwen uses grouped-query
    attention, hence ``num_key_value_heads`` rather than ``num_attention_heads``.
    """
    layers = int(getattr(config, "num_hidden_layers"))
    kv_heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    return 2 * layers * kv_heads * head_dim * sequence_length * torch.empty((), dtype=dtype).element_size()


@torch.inference_mode()
def _greedy_decode_with_prefill(
    decoder,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    max_tokens: int,
    eos_token_id: int | None,
    device: str,
) -> tuple[torch.Tensor, float, float]:
    """Greedily decode while measuring prefill once, without ``generate`` re-prefill.

    The first token comes from the prefill logits.  Subsequent tokens reuse the
    cache returned by that forward pass.  This is the measurement path; it is
    deliberately deterministic and is not a sampling API.
    """
    _sync_if_cuda(device)
    prefill_start = perf_counter()
    outputs = decoder(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        return_dict=True,
    )
    _sync_if_cuda(device)
    prefill_ms = (perf_counter() - prefill_start) * 1_000

    next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = [next_token]
    past_key_values = outputs.past_key_values
    running_mask = attention_mask
    next_position_id = position_ids[:, -1:] + 1
    _sync_if_cuda(device)
    decode_start = perf_counter()
    for _ in range(max_tokens - 1):
        if eos_token_id is not None and bool((next_token == eos_token_id).all()):
            break
        running_mask = torch.cat(
            (running_mask, torch.ones_like(next_token, dtype=running_mask.dtype)), dim=-1
        )
        outputs = decoder(
            input_ids=next_token,
            attention_mask=running_mask,
            position_ids=next_position_id,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
        next_position_id = next_position_id + 1
        generated.append(next_token)
    _sync_if_cuda(device)
    decode_ms = (perf_counter() - decode_start) * 1_000
    return torch.cat(generated, dim=-1), prefill_ms, decode_ms


@torch.inference_mode()
def build_cache(checkpoint: str, memory_jsonl: str, cache_dir: str, device: str, dtype: str) -> None:
    """Build one pre-Adapter cache per ``{"id", "context"}`` record."""
    model, _, _ = from_pretrained(checkpoint, device=device, dtype=dtype)
    records = _read_jsonl(memory_jsonl)
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = []
    seen_document_ids = set()
    for record in records:
        if "id" not in record or "context" not in record:
            raise ValueError("cache input records require string fields 'id' and 'context'")
        memory_id = document_id(record)
        if memory_id in seen_document_ids:
            raise ValueError(f"duplicate cache document_id: {memory_id}")
        seen_document_ids.add(memory_id)
        context_ids = model.encoder.embed_tokenizer.encode(record["context"], add_special_tokens=False)
        _sync_if_cuda(device)
        start = perf_counter()
        memory = model.encode_memory(context_ids, source_id=memory_id)
        _sync_if_cuda(device)
        elapsed_ms = (perf_counter() - start) * 1_000
        cache_path = root / memory_id
        save_latent_memory(memory, cache_path)
        manifest.append({
            "id": memory_id,
            "document_id": memory_id,
            "cache": str(cache_path),
            "source_tokens": len(context_ids),
            "num_latents": memory.num_latents,
            "encoder_latency_ms": elapsed_ms,
            "fingerprint": memory.fingerprint,
        })
        print(f"cached {memory_id}: {memory.num_latents} latents in {elapsed_ms:.1f} ms", flush=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _score_selector_slots(
    question_encoder: QuestionEncoder,
    selector,
    model,
    question: str,
    question_ids: torch.Tensor,
    question_mask: torch.Tensor,
    memory,
    slot_policy: str,
    slot_max_slots: int,
    device: str,
) -> tuple[torch.Tensor, int]:
    """Build query slots and score the whole memory against each of them.

    ``entity``/``sentence`` policies derive slot texts from the question and
    pool each slot with the frozen Question Encoder.  ``content_tokens``
    reuses the token-level hidden states of the original encoding and selects
    the most distinctive content tokens.  Returns ``(scores [G, M], G)``.
    """
    if slot_policy == "content_tokens":
        hidden, mask = question_encoder.forward_tokens(question_ids, question_mask)
        tokenizer = model.encoder.embed_tokenizer
        token_texts = [tokenizer.decode([int(token_id)]) for token_id in question_ids[0].tolist()]
        slots = content_token_slots(hidden[0], mask[0], token_texts, slot_max_slots)
    else:
        if slot_policy == "entity":
            slot_texts = extract_entity_slots(question)
        elif slot_policy == "sentence":
            slot_texts = split_question_sentences(question, slot_max_slots)
        else:
            raise ValueError(f"unsupported slot policy: {slot_policy}")
        encoded = model.encoder.embed_tokenizer(
            slot_texts, padding=True, return_tensors="pt", add_special_tokens=False
        )
        slots = question_encoder(
            encoded["input_ids"].to(device), encoded["attention_mask"].to(device)
        )
    if hasattr(selector, "score_slots"):
        return selector.score_slots(slots, memory), int(slots.shape[0])
    return selector.score(slots, memory), int(slots.shape[0])


@torch.inference_mode()
def answer_queries(
    checkpoint: str,
    cache_dir: str,
    queries_jsonl: str,
    out: str,
    k_decoder: int | None,
    selection_ratio: float | None,
    max_tokens: int,
    device: str,
    dtype: str,
    selector_mode: str = "cosine",
    selector_checkpoint: str | None = None,
    oracle_jsonl: str | None = None,
    random_seed: int = 4231,
    max_resident_caches: int = 1,
    neighbor_radius: int = 0,
    repeat_each_query: int = 1,
    position_policy: str = DEFAULT_POSITION_POLICY,
    utility_oracle_jsonl: str | None = None,
    compute_answer_nll: bool = False,
    max_examples: int = 0,
    coverage_rerank: bool = False,
    coverage_weight: float = 0.25,
    dynamic_k: bool = False,
    dynamic_coverage_threshold: float = 0.75,
    dynamic_budget_ratios: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    oracle_role: str | None = None,
    slot_policy: str = "none",
    slot_max_slots: int = 16,
    slot_combine: str = "union",
    slot_fusion_k: int = 60,
) -> None:
    """Run a latent-only selector baseline over cached contexts.

    ``k_decoder`` is the complete memory budget.  The feature-only MVP uses
    run-start embeddings rather than extra boundary tokens, therefore it equals
    the number of selected memory latents in this first implementation.
    """
    if max_resident_caches <= 0:
        raise ValueError("--max-resident-caches must be positive")
    if max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if neighbor_radius < 0:
        raise ValueError("--neighbor-radius must be non-negative")
    if repeat_each_query <= 0:
        raise ValueError("--repeat-each-query must be positive")
    if max_examples < 0:
        raise ValueError("--max-examples must be non-negative")
    if coverage_weight < 0:
        raise ValueError("--coverage-weight must be non-negative")
    if not 0.0 <= dynamic_coverage_threshold <= 1.0:
        raise ValueError("--dynamic-coverage-threshold must be in [0, 1]")
    if dynamic_k and k_decoder is not None:
        raise ValueError("dynamic K requires --selection-ratio as its minimum budget")
    if dynamic_k and selection_ratio is None:
        raise ValueError("dynamic K requires --selection-ratio as its minimum budget")
    if dynamic_k and selector_mode not in {"cosine", "learned"}:
        raise ValueError("dynamic K currently requires cosine or learned relevance scores")
    if coverage_rerank and selector_mode not in {"cosine", "learned"}:
        raise ValueError("coverage reranking currently requires cosine or learned relevance scores")
    if position_policy not in POSITION_POLICIES:
        raise ValueError(f"unsupported position policy: {position_policy}")
    if selector_mode not in {
        "full", "cosine", "learned", "random", "uniform", "oracle", "utility_oracle"
    }:
        raise ValueError(f"unsupported selector mode: {selector_mode}")
    if selector_mode == "learned" and not selector_checkpoint:
        raise ValueError("--selector-checkpoint is required for --selector-mode learned")
    if selector_mode != "learned" and selector_checkpoint:
        raise ValueError("--selector-checkpoint is only valid for --selector-mode learned")
    if selector_mode == "oracle" and not oracle_jsonl:
        raise ValueError("--oracle-jsonl is required for --selector-mode oracle")
    if oracle_role and selector_mode != "oracle":
        raise ValueError("--oracle-role is only valid for --selector-mode oracle")
    if selector_mode == "utility_oracle" and not utility_oracle_jsonl:
        raise ValueError("--utility-oracle-jsonl is required for utility-oracle selection")
    if neighbor_radius and selector_mode not in {"cosine", "learned"}:
        raise ValueError("--neighbor-radius is currently supported for cosine or learned selection")
    if slot_policy not in SLOT_POLICIES:
        raise ValueError(f"unsupported slot policy: {slot_policy}")
    if slot_combine not in SLOT_COMBINES:
        raise ValueError(f"unsupported slot combine: {slot_combine}")
    if slot_max_slots <= 0:
        raise ValueError("--slot-max-slots must be positive")
    if slot_fusion_k <= 0:
        raise ValueError("--slot-fusion-k must be positive")
    if slot_policy != "none":
        if selector_mode not in {"cosine", "learned"}:
            raise ValueError("slot policies currently require cosine or learned relevance scores")
        if dynamic_k or coverage_rerank:
            raise ValueError(
                "slot policies are mutually exclusive with dynamic-K and coverage rerank"
            )
        # Neighbor expansion composes with slots: it consumes the fused slot
        # ranking instead of the pooled one, keeping the exact-K contract.
    model, decoder_tokenizer, processor = from_pretrained(checkpoint, device=device, dtype=dtype)
    hidden_size = model.encoder.embed_model.config.hidden_size
    uses_question_encoder = selector_mode in {"cosine", "learned"}
    question_encoder = None
    if uses_question_encoder:
        question_encoder = QuestionEncoder(model.encoder.embed_model, hidden_size, freeze_backbone=True).to(device)
    if selector_mode == "full":
        selector = None
        positioner = None
        selector_metadata = {"selector_type": "full_lclm"}
    elif selector_mode == "learned":
        selector, positioner, selector_metadata = load_selector_checkpoint(
            selector_checkpoint, model.memory_fingerprint()
        )
        selector = selector.to(device)
    elif selector_mode == "cosine":
        selector = CosineSelector().to(device)
        positioner = None
        selector_metadata = {"selector_type": "cosine"}
    else:
        selector = None
        positioner = None
        selector_metadata = {"selector_type": selector_mode}
    if positioner is not None and positioner.net[-1].out_features != model.decoder.config.hidden_size:
        raise ValueError("selector position conditioner does not match this Decoder hidden size")
    if positioner is not None:
        positioner = positioner.to(
            device=device, dtype=next(model.decoder.parameters()).dtype,
        )
        positioner.eval()

    root = Path(cache_dir)
    oracle_labels = _load_oracle_labels(oracle_jsonl)
    oracle_roles = _load_oracle_roles(oracle_jsonl)
    oracle_fingerprints = _load_oracle_fingerprints(oracle_jsonl)
    utility_oracle = _load_utility_oracle(utility_oracle_jsonl)
    random_generator = torch.Generator(device="cpu").manual_seed(random_seed)
    results = []
    query_records = _read_jsonl(queries_jsonl)
    if max_examples:
        query_records = query_records[:max_examples]
    query_ids = [str(record.get("id")) for record in query_records]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("queries-jsonl contains duplicate query IDs")
    if repeat_each_query > 1:
        # Adjacent replicas exercise the intended GPU LRU reuse path without
        # altering the question or allowing a context reread.  This is a
        # system-amortization measurement, not an independent accuracy sample.
        query_records = [
            {**record, "query_repetition": repetition}
            for record in query_records
            for repetition in range(repeat_each_query)
        ]
    # A small LRU is enough to show the intended multi-query reuse contract
    # without retaining every document's M latent vectors on the GPU.
    resident_memories: OrderedDict[str, object] = OrderedDict()
    seen_cache_ids: set[str] = set()
    for record in query_records:
        if "id" not in record:
            raise ValueError("query records require an 'id' field")
        if "context" in record:
            raise ValueError("answer queries must not contain raw source context")
        if "selector_query" not in record and "question" not in record:
            raise ValueError("query records require 'selector_query' or 'question'")
        query_start = perf_counter()
        query_record_id = str(record["id"])
        memory_id = document_id(record)
        cache_hit = memory_id in resident_memories
        cache_first_load = memory_id not in seen_cache_ids
        if cache_hit:
            memory = resident_memories.pop(memory_id)
            resident_memories[memory_id] = memory
            cache_load_ms = 0.0
        else:
            _sync_if_cuda(device)
            cache_start = perf_counter()
            memory = load_latent_memory(root / memory_id).to(device)
            _sync_if_cuda(device)
            cache_load_ms = (perf_counter() - cache_start) * 1_000
            resident_memories[memory_id] = memory
            while len(resident_memories) > max_resident_caches:
                resident_memories.popitem(last=False)
        seen_cache_ids.add(memory_id)
        if memory.fingerprint != model.memory_fingerprint():
            raise ValueError(
                f"cache {record['id']} was built for a different Encoder configuration"
            )
        oracle_fingerprint, oracle_schema = oracle_fingerprints.get(query_record_id, (None, None))
        if oracle_fingerprint not in (None, memory.fingerprint):
            raise ValueError(f"oracle sidecar for {query_record_id} uses an incompatible cache")
        if oracle_schema not in (None, memory.schema_version):
            raise ValueError(f"oracle sidecar for {query_record_id} uses an incompatible cache schema")
        question = str(record.get("selector_query", record.get("question")))
        requested_k = _requested_k(memory.num_latents, k_decoder, selection_ratio)
        if selector_mode == "full" and requested_k != memory.num_latents:
            raise ValueError("full selector mode requires K=M (use --selection-ratio 1.0)")
        question_tokenize_ms = 0.0
        question_ids = question_mask = None
        if uses_question_encoder:
            _sync_if_cuda(device)
            preparation_start = perf_counter()
            encoded_question = model.encoder.embed_tokenizer(
                [question], padding=True, return_tensors="pt", add_special_tokens=False
            )
            question_ids = encoded_question["input_ids"].to(device)
            question_mask = encoded_question["attention_mask"].to(device)
            _sync_if_cuda(device)
            question_tokenize_ms = (perf_counter() - preparation_start) * 1_000

        selector_memory_baseline = 0
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            selector_memory_baseline = torch.cuda.memory_allocated(device)
        _sync_if_cuda(device)
        selector_start = perf_counter()
        dynamic_selection = None
        coverage_metrics: dict[str, object] = {}
        slot_metrics: dict[str, object] = {}
        if selector_mode == "full":
            selected_indices = torch.arange(memory.num_latents, device=device)
        elif selector_mode in {"cosine", "learned"}:
            assert question_encoder is not None and selector is not None
            question_vector = question_encoder(question_ids, question_mask)
            structural_groups = memory_structure_groups(memory)
            if slot_policy != "none":
                slot_scores, slot_count = _score_selector_slots(
                    question_encoder, selector, model, question, question_ids, question_mask,
                    memory, slot_policy, slot_max_slots, device,
                )
                if slot_combine == "union":
                    if neighbor_radius:
                        # Expansion consumes the slots' mean ranking as its
                        # relevance order, then restores source order under K.
                        from latent_context.selector import expand_with_neighbors
                        ranked_indices = torch.argsort(
                            slot_scores.mean(dim=0), descending=True, stable=True
                        )
                        selected_indices = expand_with_neighbors(
                            ranked_indices, memory.num_latents, requested_k, neighbor_radius
                        )
                    else:
                        selected_indices = select_slots_union(slot_scores, requested_k)
                else:
                    if slot_combine == "rank_fusion":
                        combined = reciprocal_rank_fusion(slot_scores, slot_fusion_k)
                    else:
                        combined = slot_scores.max(dim=0).values
                    if neighbor_radius:
                        from latent_context.selector import expand_with_neighbors
                        ranked_indices = torch.argsort(combined, descending=True, stable=True)
                        selected_indices = expand_with_neighbors(
                            ranked_indices, memory.num_latents, requested_k, neighbor_radius
                        )
                    else:
                        ranked = torch.argsort(combined, descending=True, stable=True)
                        selected_indices = torch.sort(ranked[:min(requested_k, combined.numel())]).values
                slot_metrics = {
                    "slot_policy": slot_policy,
                    "slot_count": slot_count,
                    "slot_combine": slot_combine,
                }
            else:
                selection = selector(question_vector, memory, requested_k)
                score_row = selection.scores[0]
                if dynamic_k:
                    dynamic_selection = select_dynamic_budget(
                        score_row,
                        minimum_ratio=float(selection_ratio),
                        budget_ratios=dynamic_budget_ratios,
                        groups=structural_groups,
                        coverage_threshold=dynamic_coverage_threshold,
                        coverage_weight=coverage_weight,
                    )
                    selected_indices = dynamic_selection.indices
                    coverage_metrics = {
                        "coverage_group_count": len(structural_groups),
                        "coverage_predicted": dynamic_selection.predicted_coverage,
                        "coverage_selected": dynamic_selection.selected_coverage,
                        "dynamic_budget_sweep": dynamic_selection.budget_sweep,
                    }
                elif coverage_rerank:
                    selected_indices = coverage_aware_greedy(
                        score_row,
                        requested_k,
                        structural_groups,
                        neighbor_radius=neighbor_radius,
                        coverage_weight=coverage_weight,
                        neighbor_weight=0.05 if neighbor_radius else 0.0,
                    )
                    coverage_metrics = {
                        "coverage_group_count": len(structural_groups),
                        "coverage_predicted": coverage_for_selection(
                            selected_indices, structural_groups
                        )["coverage"],
                        "coverage_selected": coverage_for_selection(
                            selected_indices, structural_groups
                        )["coverage"],
                    }
                elif neighbor_radius:
                    # The selector's Top-K output is source ordered.  Neighbour
                    # expansion instead needs full relevance order, then restores
                    # source order internally under the same decoder budget.
                    ranked_indices = torch.argsort(
                        selection.scores[0], descending=True, stable=True
                    )
                    from latent_context.selector import expand_with_neighbors
                    selected_indices = expand_with_neighbors(
                        ranked_indices, memory.num_latents, requested_k, neighbor_radius
                    )
                else:
                    selected_indices = selection.indices[0]
        elif selector_mode == "uniform":
            selected_indices = select_uniform_indices(memory.num_latents, requested_k, device=device)
        elif selector_mode == "random":
            selected_indices = select_random_indices(
                memory.num_latents, requested_k, generator=random_generator, device="cpu"
            ).to(device)
        elif selector_mode == "oracle":
            oracle_target = _oracle_role_indices(
                oracle_roles.get(query_record_id, {}),
                oracle_labels.get(query_record_id, []),
                oracle_role,
            )
            selected_indices = _oracle_fill_indices(
                oracle_target, memory.num_latents, requested_k, device
            )
        else:
            utility_row = utility_oracle.get(query_record_id)
            if utility_row is None:
                raise ValueError(f"missing utility-oracle selection for {query_record_id}")
            utility_document = utility_row.get("document_id")
            if utility_document is not None and str(utility_document) != memory_id:
                raise ValueError("utility-oracle sidecar points to a different document")
            if utility_row.get("cache_fingerprint") not in (None, memory.fingerprint):
                raise ValueError("utility-oracle sidecar was built from an incompatible cache")
            if utility_row.get("position_policy") not in (None, position_policy):
                raise ValueError("utility-oracle sidecar uses a different position policy")
            selected_indices = torch.tensor(
                utility_row["utility_oracle_latent_indices"], dtype=torch.long, device=device
            )
            if selected_indices.numel() != requested_k:
                raise ValueError(
                    f"utility oracle K={selected_indices.numel()} != requested K={requested_k}"
                )
            memory.select(selected_indices)
        _sync_if_cuda(device)
        selector_ms = (perf_counter() - selector_start) * 1_000
        selector_peak_memory = (
            max(torch.cuda.max_memory_allocated(device) - selector_memory_baseline, 0)
            if str(device).startswith("cuda") and torch.cuda.is_available()
            else 0
        )
        if selector_mode == "full":
            # The source-order arange is control-flow bookkeeping, not a
            # selector computation; keep the original-LCLM baseline clean.
            selector_ms = 0.0
            selector_peak_memory = 0

        selected_memory = memory.select(selected_indices)
        if not coverage_metrics and coverage_rerank:
            structural_groups = memory_structure_groups(memory)
            coverage_metrics = {
                "coverage_group_count": len(structural_groups),
                "coverage_predicted": coverage_for_selection(
                    selected_indices, structural_groups
                )["coverage"],
                "coverage_selected": coverage_for_selection(
                    selected_indices, structural_groups
                )["coverage"],
            }
        selected_span_metrics = source_span_summary(memory, selected_indices)
        adapter_memory_baseline = 0
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            adapter_memory_baseline = torch.cuda.memory_allocated(device)
        _sync_if_cuda(device)
        adapter_start = perf_counter()
        memory_embeddings = model.project_selected_latents(memory, selected_indices)
        source_length = max(int(memory.latent_end.max()), 1)
        if positioner is not None:
            memory_embeddings = positioner(memory_embeddings, selected_memory, source_length)
        _sync_if_cuda(device)
        adapter_ms = (perf_counter() - adapter_start) * 1_000
        adapter_peak_memory = (
            max(torch.cuda.max_memory_allocated(device) - adapter_memory_baseline, 0)
            if str(device).startswith("cuda") and torch.cuda.is_available()
            else 0
        )

        _sync_if_cuda(device)
        decoder_input_start = perf_counter()
        prompt = record.get("prompt")
        if prompt is None:
            user_content = f"{processor.selected_memory_marker}\n{question}\n"
            prompt = decoder_tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        if not isinstance(prompt, str) or prompt.count(processor.selected_memory_marker) != 1:
            raise ValueError(
                "query prompt must contain exactly one <|selected_memory|> marker; "
                "it must not contain raw source context"
            )
        processed = processor.process_selected_memory_batch(
            prompts=[prompt], selected_latent_counts=[int(selected_indices.numel())]
        )
        input_ids = processed["input_ids"].to(device)
        attention_mask = processed["attention_mask"].to(device)
        text_embeddings = model.decoder.get_input_embeddings()(input_ids).to(memory_embeddings.dtype)
        combined = processor.replace_memory_tokens_with_embeddings(
            inputs_embeds=text_embeddings.clone(),
            latent_embeddings=[[memory_embeddings]],
            memory_positions=processed["memory_positions"],
            latent_counts=processed["latent_counts"],
            input_ids=input_ids,
        )
        position_ids = build_decoder_position_ids(
            attention_mask,
            processed["memory_positions"],
            [selected_memory.latent_index.to(device=device, dtype=torch.long)],
            [memory.num_latents],
            position_policy,
        )
        _sync_if_cuda(device)
        decoder_input_ms = (perf_counter() - decoder_input_start) * 1_000
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            decoder_memory_baseline = torch.cuda.memory_allocated(device)
        else:
            decoder_memory_baseline = 0
        generated, prefill_ms, decode_ms = _greedy_decode_with_prefill(
            decoder=model.decoder,
            inputs_embeds=combined,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_tokens=max_tokens,
            eos_token_id=decoder_tokenizer.eos_token_id,
            device=device,
        )
        peak_memory = (
            max(torch.cuda.max_memory_allocated(device) - decoder_memory_baseline, 0)
            if str(device).startswith("cuda") and torch.cuda.is_available()
            else 0
        )
        total_ms = (perf_counter() - query_start) * 1_000
        ttft_ms = cache_load_ms + question_tokenize_ms + selector_ms + adapter_ms + decoder_input_ms + prefill_ms
        # Report the query computation independently from persistent-cache IO.
        # This is an estimate for cold requests and equals TTFT on an actual
        # LRU hit; both views are retained in the evaluator summary.
        cache_resident_ttft_ms = max(ttft_ms - cache_load_ms, 0.0)
        kv_cache_bytes = _estimate_kv_cache_bytes(
            model.decoder.config, int(attention_mask.sum().item()), next(model.decoder.parameters()).dtype
        )
        oracle_indices = set(oracle_labels.get(query_record_id, []))
        selected_set = set(selected_indices.cpu().tolist())
        role_recall: dict[str, float | None] = {}
        for role, raw_indices in oracle_roles.get(query_record_id, {}).items():
            role_indices = {int(index) for index in raw_indices}
            role_recall[role] = (
                len(selected_set & role_indices) / len(role_indices) if role_indices else None
            )
        nll_metrics: dict[str, float | None] = {}
        answer_nll_eval_ms = 0.0
        answer_token_count = None
        if compute_answer_nll:
            from inference.examples.annotate_contribution_latents import teacher_forced_nll

            _sync_if_cuda(device)
            nll_start = perf_counter()
            selected_nll, answer_token_count = teacher_forced_nll(
                model,
                processor,
                memory,
                record,
                device,
                selected_indices=selected_indices,
                position_policy=position_policy,
            )
            if selected_indices.numel() == memory.num_latents:
                full_nll = selected_nll
            else:
                full_nll, full_token_count = teacher_forced_nll(
                    model, processor, memory, record, device, position_policy=position_policy
                )
                if full_token_count != answer_token_count:
                    raise RuntimeError("selected/full answer-token counts differ")
            null_nll, null_token_count = teacher_forced_nll(
                model,
                processor,
                memory,
                record,
                device,
                selected_indices=[],
                position_policy=position_policy,
            )
            if null_token_count != answer_token_count:
                raise RuntimeError("selected/null answer-token counts differ")
            nll_metrics = answer_nll_metrics(full_nll, selected_nll, null_nll)
            _sync_if_cuda(device)
            answer_nll_eval_ms = (perf_counter() - nll_start) * 1_000
        results.append({
            "metrics_schema_version": 3,
            **{key: value for key, value in record.items() if key != "question"},
            "id": query_record_id,
            "document_id": memory_id,
            "question": question,
            "response": decoder_tokenizer.decode(generated[0], skip_special_tokens=True),
            "selected_latent_indices": selected_indices.cpu().tolist(),
            "k_decoder": int(selected_indices.numel()),
            "selection_ratio": float(selected_indices.numel() / memory.num_latents),
            "selection_policy": (
                "full" if selector_mode == "full" else
                "dynamic_coverage" if dynamic_k else
                "coverage_greedy" if coverage_rerank else
                "neighbor_expansion" if neighbor_radius else
                f"slot_{slot_combine}" if slot_policy != "none" else "relevance_topk"
            ),
            "slot_policy": slot_policy if slot_policy != "none" else None,
            "slot_count": slot_metrics.get("slot_count"),
            "slot_combine": slot_metrics.get("slot_combine"),
            "coverage_rerank": coverage_rerank,
            "coverage_weight": coverage_weight,
            "dynamic_k": dynamic_k,
            "dynamic_min_ratio": selection_ratio if dynamic_k else None,
            "dynamic_coverage_threshold": dynamic_coverage_threshold if dynamic_k else None,
            **coverage_metrics,
            **selected_span_metrics,
            "source_tokens": source_length,
            "cached_latent_count": memory.num_latents,
            "cache_load_latency_ms": cache_load_ms,
            "cache_hit": cache_hit,
            "cache_first_load": cache_first_load,
            "question_tokenize_latency_ms": question_tokenize_ms,
            "selector_latency_ms": selector_ms,
            "selector_peak_memory_bytes": selector_peak_memory,
            "adapter_latency_ms": adapter_ms,
            "adapter_peak_memory_bytes": adapter_peak_memory,
            "decoder_input_latency_ms": decoder_input_ms,
            "decoder_prefill_latency_ms": prefill_ms,
            "decoder_decode_latency_ms": decode_ms,
            "decoder_peak_memory_bytes": peak_memory,
            "decoder_kv_cache_bytes_estimate": kv_cache_bytes,
            "ttft_ms": ttft_ms,
            "cache_resident_ttft_ms": cache_resident_ttft_ms,
            "total_query_latency_ms": total_ms,
            "cache_fingerprint": memory.fingerprint,
            "position_policy": position_policy,
            "decoder_position_id_max": int(position_ids.max()),
            "virtual_full_latent_count": memory.num_latents,
            "selector_type": selector_metadata["selector_type"],
            "oracle_role": oracle_role if selector_mode == "oracle" else None,
            "utility_oracle_scope": (
                utility_oracle.get(query_record_id, {}).get("utility_oracle_scope")
                if selector_mode == "utility_oracle" else None
            ),
            "neighbor_radius": neighbor_radius,
            "oracle_label_available": query_record_id in oracle_labels,
            "oracle_latent_recall": (
                len(selected_set & oracle_indices) / len(oracle_indices) if oracle_indices else None
            ),
            "answer_latent_recall": role_recall.get("answer"),
            "all_answer_latent_recall": role_recall.get("all_answer"),
            "key_answer_joint_recall": role_recall.get("key_answer"),
            "support_latent_recall": role_recall.get("support"),
            "answer_nll_eval_only": compute_answer_nll,
            "answer_nll_eval_latency_ms": answer_nll_eval_ms,
            "teacher_answer_token_count": answer_token_count,
            **nll_metrics,
        })
    _write_jsonl(out, results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query-aware latent-only LCLM inference")
    subparsers = parser.add_subparsers(dest="command", required=True)
    cache_parser = subparsers.add_parser("cache", help="encode contexts once into Safetensors caches")
    cache_parser.add_argument("--checkpoint", required=True)
    cache_parser.add_argument("--memory-jsonl", required=True)
    cache_parser.add_argument("--cache-dir", required=True)
    cache_parser.add_argument("--device", default="cuda")
    cache_parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])

    answer_parser = subparsers.add_parser("answer", help="answer questions from latent-only caches")
    answer_parser.add_argument("--checkpoint", required=True)
    answer_parser.add_argument("--cache-dir", required=True)
    answer_parser.add_argument("--queries-jsonl", required=True)
    answer_parser.add_argument("--out", required=True)
    budget_group = answer_parser.add_mutually_exclusive_group(required=True)
    budget_group.add_argument("--k-decoder", type=int)
    budget_group.add_argument("--selection-ratio", type=float)
    answer_parser.add_argument("--max-tokens", type=int, default=128)
    answer_parser.add_argument("--device", default="cuda")
    answer_parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    answer_parser.add_argument("--selector-checkpoint",
                               help="optional Stage-A/B BiEncoder Safetensors checkpoint")
    answer_parser.add_argument("--selector-mode", default="cosine",
                               choices=[
                                   "full", "cosine", "learned", "random", "uniform", "oracle",
                                   "utility_oracle",
                               ])
    answer_parser.add_argument("--oracle-jsonl",
                               help="offline oracle sidecar; only oracle mode consumes it")
    answer_parser.add_argument(
        "--oracle-role",
        help="optional role or comma-separated role union for oracle selection "
             "(requires role labels in --oracle-jsonl)",
    )
    answer_parser.add_argument("--utility-oracle-jsonl",
                               help="offline answer-NLL oracle sidecar")
    answer_parser.add_argument("--compute-answer-nll", action="store_true",
                               help="evaluation-only full/selected/null teacher-forced NLL")
    answer_parser.add_argument("--max-examples", type=int, default=0,
                               help="0 evaluates every query")
    answer_parser.add_argument("--random-seed", type=int, default=4231)
    answer_parser.add_argument("--max-resident-caches", type=int, default=1,
                               help="GPU-resident latent-cache LRU capacity for repeated queries")
    answer_parser.add_argument("--neighbor-radius", type=int, default=0,
                               help="expand cosine/learned seeds by this many source latents under fixed K")
    answer_parser.add_argument("--coverage-rerank", action="store_true",
                               help="re-rank relevance scores with structural source-group coverage")
    answer_parser.add_argument("--coverage-weight", type=float, default=0.25,
                               help="bonus for an uncovered source group in coverage reranking")
    answer_parser.add_argument("--dynamic-k", action="store_true",
                               help="choose the smallest discrete K/M budget meeting predicted coverage")
    answer_parser.add_argument("--dynamic-coverage-threshold", type=float, default=0.75,
                               help="predicted structural coverage required by dynamic K")
    answer_parser.add_argument("--dynamic-budget-ratios", default="0.25,0.5,0.75,1.0",
                               help="comma-separated discrete K/M budgets for dynamic K")
    answer_parser.add_argument("--repeat-each-query", type=int, default=1,
                               help="adjacent repeats for measuring real latent-cache reuse")
    answer_parser.add_argument("--slot-policy", default="none", choices=list(SLOT_POLICIES),
                               help="decompose the question into multi-slot queries: "
                                    "entity enumeration, sentences, or content tokens")
    answer_parser.add_argument("--slot-max-slots", type=int, default=16,
                               help="maximum number of slots for sentence/content policies")
    answer_parser.add_argument("--slot-combine", default="union", choices=list(SLOT_COMBINES),
                               help="merge per-slot rankings by budgeted union, RRF, or max score")
    answer_parser.add_argument("--slot-fusion-k", type=int, default=60,
                               help="RRF smoothing constant for --slot-combine rank_fusion")
    answer_parser.add_argument("--position-policy", choices=POSITION_POLICIES,
                               default=DEFAULT_POSITION_POLICY,
                               help="compact or preserve full-memory RoPE positions across sparse gaps")
    args = parser.parse_args()
    if args.command == "cache":
        build_cache(args.checkpoint, args.memory_jsonl, args.cache_dir, args.device, args.dtype)
    else:
        try:
            dynamic_budget_ratios = tuple(
                float(value) for value in args.dynamic_budget_ratios.split(",") if value.strip()
            )
        except ValueError as exc:
            raise ValueError("--dynamic-budget-ratios must be comma-separated numbers") from exc
        answer_queries(
            args.checkpoint, args.cache_dir, args.queries_jsonl, args.out,
            args.k_decoder, args.selection_ratio, args.max_tokens, args.device, args.dtype,
            args.selector_mode, args.selector_checkpoint,
            args.oracle_jsonl, args.random_seed, args.max_resident_caches,
            args.neighbor_radius, args.repeat_each_query, args.position_policy,
            args.utility_oracle_jsonl, args.compute_answer_nll, args.max_examples,
            args.coverage_rerank, args.coverage_weight, args.dynamic_k,
            args.dynamic_coverage_threshold, dynamic_budget_ratios,
            args.oracle_role,
            args.slot_policy, args.slot_max_slots, args.slot_combine, args.slot_fusion_k,
        )


if __name__ == "__main__":
    main()
