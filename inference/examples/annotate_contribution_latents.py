"""Annotate cached latents with full-latent teacher-forced contributions.

The script is an offline teacher pass for selector training.  It consumes the
raw context only indirectly through a pre-built ``LatentMemory`` cache and the
query sidecar's teacher answer.  The emitted sidecar contains no source text
and is never read by answer-time inference.

For a query with pre-Adapter latents ``z`` and answer-only mean NLL ``L``, the
default target is the first-order magnitude ``abs(z * dL/dz)`` reduced over the
latent embedding dimension.  This gives one non-negative contribution score
per latent while keeping the selector's existing pre-Adapter retrieval space.

Example::

    python -m inference.examples.annotate_contribution_latents \
      --checkpoint artifacts/models/0.6b-4b-LCLM-8x \
      --cache-dir artifacts/data/selector_train_mixed_4to32k/cache \
      --queries-jsonl artifacts/data/selector_train_mixed_4to32k/queries.jsonl \
      --out artifacts/data/selector_train_mixed_4to32k/contributions-zgrad.jsonl \
      --max-examples 8 --max-target-tokens 32
"""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from latent_context import (
    DEFAULT_POSITION_POLICY,
    POSITION_POLICIES,
    build_decoder_position_ids,
    document_id,
    from_pretrained,
)
from latent_context.selector import LatentMemory, load_latent_memory


CONTRIBUTION_SCHEMA_VERSION = 2
TARGET_DEFINITION_PREFIX = "zgrad_pre_adapter_abs_l2"
TARGET_DEFINITION = f"{TARGET_DEFINITION_PREFIX}_{DEFAULT_POSITION_POLICY}_v2"


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_teacher_target(answers: Iterable[Any], separator: str = ", ") -> str:
    """Build the deterministic answer string used by the teacher-forced pass."""
    values = [str(answer) for answer in answers]
    if not values:
        raise ValueError("at least one answer is required for contribution annotation")
    if not isinstance(separator, str):
        raise TypeError("answer separator must be a string")
    return separator.join(values)


def compute_zgrad_contributions(
    latents: torch.Tensor, gradients: torch.Tensor,
) -> torch.Tensor:
    """Return one ``abs(z * dL/dz)`` L2 score per latent.

    Both tensors must have shape ``[M, D]``.  Detaching here makes it explicit
    that contribution sidecars are labels, not part of the selector graph.
    """
    if latents.ndim != 2 or gradients.shape != latents.shape:
        raise ValueError("latents and gradients must both have shape [M, D]")
    contributions = (latents.detach() * gradients.detach()).abs().norm(dim=-1)
    return torch.nan_to_num(contributions, nan=0.0, posinf=0.0, neginf=0.0).float()


def compute_latent_contributions(latents: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
    """Compatibility alias for the default contribution target."""
    return compute_zgrad_contributions(latents, gradients)


def rank_contributions(contributions: torch.Tensor) -> torch.Tensor:
    """Return one-based ranks (1 is most contributing), with stable ties."""
    if contributions.ndim != 1 or contributions.numel() == 0:
        raise ValueError("contributions must be a non-empty vector")
    if not bool(torch.isfinite(contributions).all()):
        raise ValueError("contributions must be finite")
    order = torch.argsort(contributions, descending=True, stable=True)
    ranks = torch.empty_like(order, dtype=torch.long)
    ranks[order] = torch.arange(1, order.numel() + 1, device=order.device)
    return ranks


def _answer_only_loss_from_hidden(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    lm_head: torch.nn.Module,
) -> tuple[torch.Tensor, int]:
    """Compute answer-only CE while materializing logits for answer tokens only."""
    if hidden_states.ndim != 3 or labels.ndim != 2:
        raise ValueError("hidden_states must be [B, S, H] and labels must be [B, S]")
    if hidden_states.shape[:2] != labels.shape:
        raise ValueError("hidden_states and labels sequence shapes must match")
    labels = labels.to(hidden_states.device)
    shift_hidden = hidden_states[:, :-1, :]
    shift_labels = labels[:, 1:]
    answer_mask = shift_labels.ne(-100)
    token_count = int(answer_mask.sum())
    if token_count <= 0:
        raise ValueError("teacher target produced no answer tokens")
    selected_hidden = shift_hidden[answer_mask]
    selected_labels = shift_labels[answer_mask]
    logits = lm_head(selected_hidden)
    loss_sum = F.cross_entropy(
        logits.float(), selected_labels.to(device=logits.device), reduction="sum"
    )
    return loss_sum, token_count


def _decoder_hidden_and_head(
    decoder: torch.nn.Module,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    activation_checkpointing: bool = False,
) -> tuple[torch.Tensor, torch.nn.Module]:
    """Run the Decoder backbone and return hidden states plus its LM head.

    Activation checkpointing is used only by the differentiable annotation
    pass.  The non-reentrant implementation is required because annotation
    obtains the latent gradient with ``torch.autograd.grad``.  Qwen3 exposes
    checkpoint-aware decoder layers; those layers are enabled before this
    backbone call so recomputation is performed one layer at a time.
    """
    lm_head = getattr(decoder, "lm_head", None)
    if lm_head is None and hasattr(decoder, "get_output_embeddings"):
        lm_head = decoder.get_output_embeddings()
    if lm_head is None:
        raise AttributeError("decoder does not expose an output/lm head")

    backbone = getattr(decoder, "model", None)
    if backbone is not None:
        def run_backbone(embeds: torch.Tensor) -> torch.Tensor:
            try:
                output = backbone(
                    input_ids=None,
                    inputs_embeds=embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            except TypeError:
                # Small mock backbones and older Transformers omit some keyword
                # arguments; the standard path above remains the production one.
                output = backbone(
                    inputs_embeds=embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None and isinstance(output, (tuple, list)):
                hidden = output[0]
            if hidden is None:
                raise RuntimeError("decoder backbone did not return hidden states")
            return hidden

        hidden_states = run_backbone(inputs_embeds)
    else:
        output = decoder(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=None,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = getattr(output, "hidden_states", None)
        if isinstance(hidden_states, (tuple, list)) and hidden_states:
            hidden_states = hidden_states[-1]
        if hidden_states is None:
            hidden_states = getattr(output, "last_hidden_state", None)
    if hidden_states is None:
        raise RuntimeError("decoder forward did not return hidden states")
    return hidden_states, lm_head


def _enable_decoder_layer_checkpointing(decoder: torch.nn.Module) -> int:
    """Enable frozen Decoder layer checkpoints without changing dropout mode."""
    backbone = getattr(decoder, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        return 0
    checkpoint_fn = partial(checkpoint, use_reentrant=False)
    enabled = 0
    for layer in layers:
        if not hasattr(layer, "gradient_checkpointing"):
            continue
        layer.gradient_checkpointing = True
        layer._gradient_checkpointing_func = checkpoint_fn
        # GradientCheckpointingLayer checks its own training flag.  Keep the
        # attention/MLP children in eval mode so this does not enable dropout.
        layer.training = True
        enabled += 1
    return enabled


def _replace_single_region(
    text_embeddings: torch.Tensor,
    projected_latents: torch.Tensor,
    memory_positions: list[list[tuple[int, int]]],
    expected_count: int,
) -> torch.Tensor:
    """Inject projected latents without an in-place assignment on the graph."""
    if text_embeddings.ndim != 3 or text_embeddings.shape[0] != 1:
        raise ValueError("contribution annotation currently expects a batch of one")
    if projected_latents.ndim != 2 or projected_latents.shape[0] != expected_count:
        raise ValueError("projected latent count does not match the selected-memory marker")
    if len(memory_positions) != 1 or len(memory_positions[0]) != 1:
        raise ValueError("teacher prompt must contain exactly one memory region")
    start, end = memory_positions[0][0]
    if end - start != expected_count or not (0 <= start <= end <= text_embeddings.shape[1]):
        raise ValueError("invalid tokenized memory position range")
    return torch.cat(
        (text_embeddings[:, :start], projected_latents.unsqueeze(0), text_embeddings[:, end:]),
        dim=1,
    )


def _mask_answer_tokens(labels: torch.Tensor, max_target_tokens: int | None) -> torch.Tensor:
    if max_target_tokens is None:
        return labels
    if max_target_tokens <= 0:
        raise ValueError("max_target_tokens must be positive when provided")
    masked = labels.clone()
    answer_positions = masked[0].ne(-100).nonzero(as_tuple=False).flatten()
    if answer_positions.numel() > max_target_tokens:
        masked[0, answer_positions[max_target_tokens:]] = -100
    return masked


def _teacher_target(query: dict[str, Any], answer_separator: str) -> str:
    target = query.get("teacher_target")
    if target is None:
        answers = list(dict.fromkeys(str(answer) for answer in query.get("answers", [])))
        if query.get("scoring_function") == "ruler_string_match_part":
            target = answers[0] if answers else ""
        else:
            target = build_teacher_target(answers, answer_separator)
    if not isinstance(target, str) or not target:
        raise ValueError(f"query {query.get('id')} has an empty teacher target")
    return target


def teacher_forced_nll(
    model,
    processor,
    memory: LatentMemory,
    query: dict[str, Any],
    device: str | torch.device,
    max_target_tokens: int | None = None,
    answer_separator: str = ", ",
    selected_indices: Iterable[int] | torch.Tensor | None = None,
    position_policy: str = DEFAULT_POSITION_POLICY,
) -> tuple[float, int]:
    """Evaluate answer-only NLL for a subset under an explicit position policy.

    ``memory`` is always the complete cached memory. ``selected_indices`` may
    be empty for the no-memory control; omitted indices select all latents.
    """
    prompt = query.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"query {query.get('id')} is missing prompt")
    target = _teacher_target(query, answer_separator)
    indices = (
        torch.arange(memory.num_latents, dtype=torch.long, device=memory.latents.device)
        if selected_indices is None
        else torch.as_tensor(selected_indices, dtype=torch.long, device=memory.latents.device)
    )
    if indices.ndim != 1:
        raise ValueError("selected_indices must have shape [K]")
    selected_memory = memory.select(indices) if indices.numel() else None
    selected_count = int(indices.numel())
    processed = processor.process_selected_memory_batch(
        prompts=[prompt],
        selected_latent_counts=[selected_count],
        targets=[target],
        padding="longest",
        truncation=False,
        return_tensors="pt",
        allow_empty_memory=True,
    )
    input_ids = processed["input_ids"].to(device)
    attention_mask = processed["attention_mask"].to(device)
    labels = _mask_answer_tokens(processed["labels"].to(device), max_target_tokens)
    adapter = model.adapter
    adapter_parameter = next(adapter.parameters(), None)
    adapter_device = adapter_parameter.device if adapter_parameter is not None else input_ids.device
    adapter_dtype = adapter_parameter.dtype if adapter_parameter is not None else torch.float32
    decoder = model.decoder
    with torch.no_grad():
        decoder_parameter = next(decoder.parameters(), None)
        compute_dtype = decoder_parameter.dtype if decoder_parameter is not None else adapter_dtype
        text_embeddings = decoder.get_input_embeddings()(input_ids).to(compute_dtype)
        if selected_memory is None:
            projected = text_embeddings.new_empty((0, text_embeddings.shape[-1]))
            original_indices = torch.empty(0, dtype=torch.long, device=input_ids.device)
        else:
            latents = selected_memory.latents.to(device=adapter_device, dtype=adapter_dtype)
            projected = adapter(latents).to(device=text_embeddings.device, dtype=compute_dtype)
            original_indices = selected_memory.latent_index.to(input_ids.device, dtype=torch.long)
        combined = _replace_single_region(
            text_embeddings,
            projected,
            processed["memory_positions"],
            selected_count,
        )
        position_ids = build_decoder_position_ids(
            attention_mask,
            processed["memory_positions"],
            [original_indices],
            [memory.num_latents],
            position_policy,
        )
        hidden_states, lm_head = _decoder_hidden_and_head(
            decoder,
            combined,
            attention_mask,
            position_ids=position_ids,
        )
        loss_sum, token_count = _answer_only_loss_from_hidden(hidden_states, labels, lm_head)
    return float((loss_sum / token_count).cpu()), token_count


def leave_one_out_nll_deltas(
    model,
    processor,
    memory: LatentMemory,
    query: dict[str, Any],
    device: str | torch.device,
    indices: Iterable[int] | None = None,
    max_target_tokens: int | None = None,
    answer_separator: str = ", ",
    position_policy: str = DEFAULT_POSITION_POLICY,
) -> tuple[list[int], list[float], float]:
    """Measure NLL increase after removing each requested latent.

    This is intentionally an offline diagnostic.  It is quadratic in the
    number of selected ablations and should be used on a small validation
    subset, not during online answer generation.
    """
    full_nll, _ = teacher_forced_nll(
        model,
        processor,
        memory,
        query,
        device,
        max_target_tokens,
        answer_separator,
        position_policy=position_policy,
    )
    candidates = list(range(memory.num_latents) if indices is None else indices)
    deltas: list[float] = []
    valid_indices: list[int] = []
    for index in candidates:
        if index < 0 or index >= memory.num_latents:
            raise IndexError(f"leave-one-out index {index} is out of range")
        if memory.num_latents <= 1:
            continue
        keep = torch.arange(memory.num_latents, device=memory.latents.device)
        keep = keep[keep != int(index)]
        ablated_nll, _ = teacher_forced_nll(
            model,
            processor,
            memory,
            query,
            device,
            max_target_tokens,
            answer_separator,
            selected_indices=keep,
            position_policy=position_policy,
        )
        valid_indices.append(int(index))
        deltas.append(float(ablated_nll - full_nll))
    return valid_indices, deltas, full_nll


def _rankdata(values: Iterable[float]) -> torch.Tensor:
    values_tensor = torch.tensor(list(values), dtype=torch.float64)
    if values_tensor.numel() == 0:
        return values_tensor
    order = torch.argsort(values_tensor, stable=True)
    ranks = torch.empty_like(values_tensor)
    cursor = 0
    while cursor < order.numel():
        end = cursor + 1
        while end < order.numel() and values_tensor[order[end]] == values_tensor[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + end - 1) / 2.0 + 1.0
        cursor = end
    return ranks


def pearson_correlation(first: Iterable[float], second: Iterable[float]) -> float:
    x = torch.tensor(list(first), dtype=torch.float64)
    y = torch.tensor(list(second), dtype=torch.float64)
    if x.numel() != y.numel() or x.numel() < 2:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.sqrt((x.square().sum()) * (y.square().sum()))
    if float(denominator) == 0.0:
        return 0.0
    return float((x * y).sum() / denominator)


def spearman_correlation(first: Iterable[float], second: Iterable[float]) -> float:
    return pearson_correlation(_rankdata(first).tolist(), _rankdata(second).tolist())


def annotate_query(
    model,
    processor,
    memory: LatentMemory,
    query: dict[str, Any],
    device: str | torch.device,
    max_target_tokens: int | None = None,
    answer_separator: str = ", ",
    activation_checkpointing: bool = False,
    position_policy: str = DEFAULT_POSITION_POLICY,
) -> dict[str, Any]:
    """Run one differentiable full-latent teacher pass and emit its label row."""
    prompt = query.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"query {query.get('id')} is missing prompt")
    if "context" in query:
        raise ValueError("queries-jsonl must not contain raw context")
    target = _teacher_target(query, answer_separator)

    processed = processor.process_selected_memory_batch(
        prompts=[prompt],
        selected_latent_counts=[memory.num_latents],
        targets=[target],
        padding="longest",
        truncation=False,
        return_tensors="pt",
    )
    input_ids = processed["input_ids"].to(device)
    attention_mask = processed["attention_mask"].to(device)
    labels = _mask_answer_tokens(processed["labels"].to(device), max_target_tokens)

    adapter = model.adapter
    adapter_parameter = next(adapter.parameters(), None)
    adapter_device = adapter_parameter.device if adapter_parameter is not None else input_ids.device
    adapter_dtype = adapter_parameter.dtype if adapter_parameter is not None else torch.float32
    pre_adapter_latents = memory.latents.to(device=adapter_device, dtype=adapter_dtype).detach()
    pre_adapter_latents.requires_grad_(True)
    projected = adapter(pre_adapter_latents)

    decoder = model.decoder
    decoder_parameter = next(decoder.parameters(), None)
    compute_dtype = decoder_parameter.dtype if decoder_parameter is not None else torch.float32
    text_embeddings = decoder.get_input_embeddings()(input_ids).to(compute_dtype)
    combined = _replace_single_region(
        text_embeddings,
        projected.to(device=text_embeddings.device, dtype=compute_dtype),
        processed["memory_positions"],
        memory.num_latents,
    )
    position_ids = build_decoder_position_ids(
        attention_mask,
        processed["memory_positions"],
        [memory.latent_index.to(input_ids.device, dtype=torch.long)],
        [memory.num_latents],
        position_policy,
    )
    hidden_states, lm_head = _decoder_hidden_and_head(
        decoder,
        combined,
        attention_mask,
        position_ids=position_ids,
        activation_checkpointing=activation_checkpointing,
    )
    loss_sum, token_count = _answer_only_loss_from_hidden(hidden_states, labels, lm_head)
    mean_loss = loss_sum / token_count
    gradients = torch.autograd.grad(mean_loss, pre_adapter_latents, retain_graph=False)[0]
    contributions = compute_zgrad_contributions(pre_adapter_latents, gradients)
    ranks = rank_contributions(contributions)
    return {
        "id": str(query["id"]),
        "document_id": document_id(query),
        "task": query.get("task"),
        "latent_contributions": contributions.cpu().tolist(),
        "latent_contribution_ranks": ranks.cpu().tolist(),
        "teacher_nll": float(mean_loss.detach().cpu()),
        "teacher_answer_token_count": token_count,
        "target_definition": f"{TARGET_DEFINITION_PREFIX}_{position_policy}_v2",
        "contribution_schema_version": CONTRIBUTION_SCHEMA_VERSION,
        "position_policy": position_policy,
        "attribution_input_layout": "full_latent_teacher_forced",
        "cache_fingerprint": memory.fingerprint,
        "cache_schema_version": memory.schema_version,
    }


def _load_memory(cache_dir: Path, record_id: str, device: str) -> LatentMemory:
    memory = load_latent_memory(cache_dir / record_id, device=device)
    return memory


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate full-latent teacher contributions")
    parser.add_argument("--checkpoint", required=True, help="local LCLM checkpoint")
    parser.add_argument("--cache-dir", required=True, help="pre-Adapter LatentMemory directory")
    parser.add_argument("--queries-jsonl", required=True, help="latent-only queries with answers")
    parser.add_argument("--out", required=True, help="contribution sidecar JSONL")
    parser.add_argument("--start-index", type=int, default=0,
                        help="zero-based query offset before applying --max-examples")
    parser.add_argument("--max-examples", type=int, default=0, help="0 annotates all queries")
    parser.add_argument("--max-target-tokens", type=int, default=0,
                        help="0 keeps the complete teacher target")
    parser.add_argument("--answer-separator", default=", ")
    parser.add_argument("--position-policy", choices=POSITION_POLICIES,
                        default=DEFAULT_POSITION_POLICY)
    parser.add_argument("--leave-one-out", action="store_true",
                        help="also run offline latent ablations for diagnostics")
    parser.add_argument("--activation-checkpointing", action="store_true",
                        help="recompute Decoder activations during backward to reduce peak GPU memory")
    parser.add_argument("--max-loo-latents", type=int, default=32,
                        help="0 tests every latent; otherwise tests the top contribution latents")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args()
    if args.start_index < 0 or args.max_examples < 0 or args.max_target_tokens < 0 or args.max_loo_latents < 0:
        raise ValueError("start-index, max-examples, max-target-tokens, and max-loo-latents must be non-negative")

    queries = _read_jsonl(args.queries_jsonl)
    if args.start_index >= len(queries):
        raise ValueError(f"start-index {args.start_index} is outside the {len(queries)} query rows")
    queries = queries[args.start_index:]
    if args.max_examples:
        queries = queries[:args.max_examples]
    if not queries:
        raise ValueError("queries-jsonl contains no examples")

    model, _, processor = from_pretrained(args.checkpoint, device=args.device, dtype=args.dtype)
    model.eval()
    # Only the input latent requires a gradient.  Freezing all model weights
    # substantially reduces the retained graph for the 4B Decoder teacher.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    checkpointed_layers = _enable_decoder_layer_checkpointing(model.decoder) if args.activation_checkpointing else 0
    if args.activation_checkpointing:
        print(json.dumps({"activation_checkpointing": True, "checkpointed_decoder_layers": checkpointed_layers}), flush=True)
    fingerprint = model.memory_fingerprint()
    cache_dir = Path(args.cache_dir)
    rows = []
    skipped = 0
    for query in queries:
        record_id = str(query.get("id"))
        memory_id = document_id(query)
        memory = _load_memory(cache_dir, memory_id, args.device)
        if memory.fingerprint != fingerprint:
            raise ValueError(f"cache {record_id} was built by an incompatible Encoder")
        row = annotate_query(
            model,
            processor,
            memory,
            query,
            args.device,
            max_target_tokens=args.max_target_tokens or None,
            answer_separator=args.answer_separator,
            activation_checkpointing=args.activation_checkpointing,
            position_policy=args.position_policy,
        )
        if args.leave_one_out:
            contributions = torch.tensor(row["latent_contributions"], dtype=torch.float32)
            order = torch.argsort(contributions, descending=True, stable=True)
            limit = memory.num_latents if args.max_loo_latents == 0 else min(
                args.max_loo_latents, memory.num_latents
            )
            loo_indices, loo_deltas, loo_full_nll = leave_one_out_nll_deltas(
                model,
                processor,
                memory,
                query,
                args.device,
                indices=order[:limit].tolist(),
                max_target_tokens=args.max_target_tokens or None,
                answer_separator=args.answer_separator,
                position_policy=args.position_policy,
            )
            selected_contributions = [row["latent_contributions"][index] for index in loo_indices]
            row.update({
                "leave_one_out_indices": loo_indices,
                "leave_one_out_nll_deltas": loo_deltas,
                "leave_one_out_full_nll": loo_full_nll,
                "contribution_loo_pearson": pearson_correlation(selected_contributions, loo_deltas),
                "contribution_loo_spearman": spearman_correlation(selected_contributions, loo_deltas),
            })
        rows.append(row)
        print(json.dumps({
            "id": record_id,
            "teacher_nll": row["teacher_nll"],
            "answer_tokens": row["teacher_answer_token_count"],
            "num_latents": len(row["latent_contributions"]),
        }), flush=True)

    _write_jsonl(args.out, rows)
    print(json.dumps({
        "out": str(args.out),
        "records": len(rows),
        "skipped": skipped,
        "target_definition": f"{TARGET_DEFINITION_PREFIX}_{args.position_policy}_v2",
        "contribution_schema_version": CONTRIBUTION_SCHEMA_VERSION,
        "position_policy": args.position_policy,
        "cache_fingerprint": fingerprint,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
