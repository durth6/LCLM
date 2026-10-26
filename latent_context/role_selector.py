"""Explicit multi-head role-aware latent selector.

The ordinary BiEncoder emits one relevance score per latent.  This module
keeps the answer-time contract latent-only while learning separate
query-conditioned evidence heads (relevance, key, answer and support).  A
hard source-ordered selection combines the heads with minimum evidence quotas;
the quotas are structural selector behavior, never oracle labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .selector import LatentMemory, SelectionOutput


ROLE_NAMES = ("relevance", "key", "answer", "support")


class RoleAwareSelector(nn.Module):
    """Multi-head BiEncoder with query-conditioned evidence fusion."""

    selector_type = "role_aware_biencoder"

    def __init__(
        self,
        input_dim: int,
        retrieval_dim: int = 128,
        normalize: bool = True,
        min_evidence_fraction: float = 0.08,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or retrieval_dim <= 0:
            raise ValueError("input_dim and retrieval_dim must be positive")
        if not 0.0 <= min_evidence_fraction <= 0.5:
            raise ValueError("min_evidence_fraction must be in [0, 0.5]")
        self.input_dim = int(input_dim)
        self.retrieval_dim = int(retrieval_dim)
        self.normalize = bool(normalize)
        self.min_evidence_fraction = float(min_evidence_fraction)
        self.query_projection = nn.ModuleDict({
            role: nn.Linear(input_dim, retrieval_dim, bias=False)
            for role in ROLE_NAMES
        })
        self.latent_projection = nn.ModuleDict({
            role: nn.Linear(input_dim, retrieval_dim, bias=False)
            for role in ROLE_NAMES
        })
        self.log_temperature = nn.ParameterDict({
            role: nn.Parameter(torch.zeros(())) for role in ROLE_NAMES
        })
        self.gate = nn.Linear(input_dim, len(ROLE_NAMES), bias=True)
        nn.init.zeros_(self.gate.bias)

    def role_scores(self, question_vectors: torch.Tensor, memory: LatentMemory) -> dict[str, torch.Tensor]:
        if question_vectors.ndim != 2:
            raise ValueError("question_vectors must have shape [B, D]")
        output: dict[str, torch.Tensor] = {}
        for role in ROLE_NAMES:
            dtype = self.query_projection[role].weight.dtype
            q = self.query_projection[role](question_vectors.to(dtype=dtype))
            z = self.latent_projection[role](memory.latents.to(q.device, dtype=dtype))
            if self.normalize:
                q = F.normalize(q, dim=-1)
                z = F.normalize(z, dim=-1)
            temperature = self.log_temperature[role].exp().clamp(min=0.01, max=100.0)
            output[role] = (q @ z.T) / temperature
        return output

    def gate_weights(self, question_vectors: torch.Tensor) -> torch.Tensor:
        dtype = self.gate.weight.dtype
        return F.softmax(self.gate(question_vectors.to(dtype=dtype)), dim=-1)

    def score(self, question_vectors: torch.Tensor, memory: LatentMemory) -> torch.Tensor:
        role_scores = self.role_scores(question_vectors, memory)
        gates = self.gate_weights(question_vectors).to(next(iter(role_scores.values())).dtype)
        fused = torch.zeros_like(role_scores[ROLE_NAMES[0]])
        for index, role in enumerate(ROLE_NAMES):
            fused = fused + gates[:, index:index + 1] * role_scores[role]
        return fused

    @torch.no_grad()
    def _diversified_indices(
        self,
        role_scores: dict[str, torch.Tensor],
        fused_scores: torch.Tensor,
        k: int,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        """Select exact K using predicted role heads, then restore source order."""
        batch, memory_length = fused_scores.shape
        actual_k = min(int(k), memory_length)
        output = []
        for batch_index in range(batch):
            gate = gates[batch_index].float()
            # Evidence heads get a small minimum quota when capacity permits;
            # the remaining slots follow the learned query-conditioned gate.
            quotas = torch.floor(gate * actual_k).long()
            if actual_k >= len(ROLE_NAMES):
                minimum = max(1, int(actual_k * self.min_evidence_fraction))
                for role_index in range(1, len(ROLE_NAMES)):
                    quotas[role_index] = max(int(quotas[role_index]), minimum)
            while int(quotas.sum()) > actual_k:
                candidates = [i for i in range(len(ROLE_NAMES)) if quotas[i] > 0]
                drop = min(candidates, key=lambda i: (float(gate[i]), i))
                quotas[drop] -= 1
            while int(quotas.sum()) < actual_k:
                add = int(torch.argmax(gate).item())
                quotas[add] += 1
            selected: set[int] = set()
            for role_index, role in enumerate(ROLE_NAMES):
                quota = int(quotas[role_index])
                if quota <= 0:
                    continue
                ranked = torch.argsort(role_scores[role][batch_index], descending=True, stable=True)
                selected.update(int(value) for value in ranked[:quota].tolist())
            if len(selected) < actual_k:
                ranked = torch.argsort(fused_scores[batch_index], descending=True, stable=True)
                for value in ranked.tolist():
                    selected.add(int(value))
                    if len(selected) >= actual_k:
                        break
            if len(selected) > actual_k:
                ranked = torch.argsort(fused_scores[batch_index], descending=True, stable=True)
                selected = set(sorted(selected, key=lambda value: (-float(fused_scores[batch_index, value]), value))[:actual_k])
            output.append(torch.tensor(sorted(selected), dtype=torch.long, device=fused_scores.device))
        return torch.stack(output)

    def forward(self, question_vectors: torch.Tensor, memory: LatentMemory, k: int) -> SelectionOutput:
        role_scores = self.role_scores(question_vectors, memory)
        gates = self.gate_weights(question_vectors)
        fused = torch.zeros_like(role_scores[ROLE_NAMES[0]])
        for index, role in enumerate(ROLE_NAMES):
            fused = fused + gates[:, index:index + 1].to(fused.dtype) * role_scores[role]
        indices = self._diversified_indices(role_scores, fused, k, gates)
        return SelectionOutput(
            indices=indices,
            scores=fused,
            selected_count=torch.full(
                (question_vectors.shape[0],), indices.shape[1], dtype=torch.int32,
                device=question_vectors.device,
            ),
        )


def save_role_selector_checkpoint(
    checkpoint_dir: str | Path,
    selector: RoleAwareSelector,
    base_memory_fingerprint: str,
    training_metadata: dict[str, Any] | None = None,
) -> None:
    from safetensors.torch import save_file

    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    save_file(
        {f"role_selector.{name}": value.detach().cpu().contiguous()
         for name, value in selector.state_dict().items()},
        str(path / "role_selector.safetensors"),
    )
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "selector_type": RoleAwareSelector.selector_type,
        "input_dim": selector.input_dim,
        "retrieval_dim": selector.retrieval_dim,
        "normalize": selector.normalize,
        "min_evidence_fraction": selector.min_evidence_fraction,
        "roles": list(ROLE_NAMES),
        "base_memory_fingerprint": base_memory_fingerprint,
    }
    if training_metadata:
        metadata.update(training_metadata)
    (path / "role_selector_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_role_selector_checkpoint(
    checkpoint_dir: str | Path,
    expected_memory_fingerprint: str,
) -> tuple[RoleAwareSelector, dict[str, Any]]:
    from safetensors.torch import load_file

    path = Path(checkpoint_dir)
    metadata = json.loads((path / "role_selector_metadata.json").read_text())
    if metadata.get("schema_version") != 1 or metadata.get("selector_type") != RoleAwareSelector.selector_type:
        raise ValueError("unsupported role-aware selector checkpoint")
    if metadata.get("base_memory_fingerprint") != expected_memory_fingerprint:
        raise ValueError("role-aware selector was trained for a different latent encoder")
    selector = RoleAwareSelector(
        int(metadata["input_dim"]),
        int(metadata["retrieval_dim"]),
        bool(metadata.get("normalize", True)),
        float(metadata.get("min_evidence_fraction", 0.08)),
    )
    tensors = load_file(str(path / "role_selector.safetensors"))
    selector.load_state_dict({
        name.removeprefix("role_selector."): tensor
        for name, tensor in tensors.items() if name.startswith("role_selector.")
    }, strict=True)
    return selector, metadata
