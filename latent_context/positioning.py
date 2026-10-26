"""Decoder position policies for sparse latent-memory inputs."""

from __future__ import annotations

from collections.abc import Sequence

import torch


POSITION_POLICIES = ("compact", "full_layout")
DEFAULT_POSITION_POLICY = "full_layout"


def build_decoder_position_ids(
    attention_mask: torch.Tensor,
    memory_positions: Sequence[Sequence[tuple[int, int]]],
    selected_latent_indices: Sequence[torch.Tensor | Sequence[int]],
    full_latent_counts: Sequence[int],
    policy: str = DEFAULT_POSITION_POLICY,
) -> torch.Tensor:
    """Build RoPE positions for compact tensors representing sparse memory.

    ``full_layout`` assigns each selected memory embedding the position it had
    in the full latent sequence. Tokens after the memory region are shifted by
    ``M-K``, so the question, answer, and generated continuation retain their
    full-memory virtual positions even though omitted latents consume no KV or
    attention compute. Causal visibility still follows compact tensor order.
    """
    if policy not in POSITION_POLICIES:
        raise ValueError(f"position policy must be one of {POSITION_POLICIES}, got {policy!r}")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [B, S]")
    batch_size, sequence_length = attention_mask.shape
    if not (
        len(memory_positions) == len(selected_latent_indices) == len(full_latent_counts) == batch_size
    ):
        raise ValueError("position metadata must contain one entry per batch row")

    position_ids = torch.arange(
        sequence_length, dtype=torch.long, device=attention_mask.device
    ).unsqueeze(0).expand(batch_size, -1).clone()
    if policy == "compact":
        return position_ids

    for batch_index, (regions, raw_indices, raw_full_count) in enumerate(
        zip(memory_positions, selected_latent_indices, full_latent_counts)
    ):
        if len(regions) != 1:
            raise ValueError("full_layout currently requires exactly one memory region per query")
        start, end = regions[0]
        if not (0 <= start <= end <= sequence_length):
            raise ValueError(f"invalid memory region [{start}, {end}) for sequence length {sequence_length}")
        indices = torch.as_tensor(raw_indices, dtype=torch.long, device=attention_mask.device)
        if indices.ndim != 1 or indices.numel() != end - start:
            raise ValueError("selected latent indices must match the compact memory-region length")
        full_count = int(raw_full_count)
        if full_count < indices.numel() or full_count < 0:
            raise ValueError("full latent count must be non-negative and at least K")
        if indices.numel():
            if int(indices.min()) < 0 or int(indices.max()) >= full_count:
                raise IndexError("selected latent index is outside the full-memory layout")
            if indices.numel() > 1 and bool((indices[1:] <= indices[:-1]).any()):
                raise ValueError("selected latent indices must be strictly source ordered")
            position_ids[batch_index, start:end] = start + indices
        position_ids[batch_index, end:] += full_count - indices.numel()

        active = position_ids[batch_index][attention_mask[batch_index].bool()]
        if active.numel() > 1 and bool((active[1:] <= active[:-1]).any()):
            raise ValueError("full-layout active position IDs must be strictly increasing")
    return position_ids
