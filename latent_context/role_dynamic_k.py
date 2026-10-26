"""Validation-calibrated dynamic budgets for the role-aware latent selector."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch

from .role_selector import ROLE_NAMES, RoleAwareSelector
from .selector import DynamicSelectionOutput, LatentMemory


PREDICTED_ROLES = ("key", "answer", "support")


def fused_role_scores(
    selector: RoleAwareSelector,
    question_vectors: torch.Tensor,
    memory: LatentMemory,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Compute role heads, query gates and their fused relevance score once."""
    role_scores = selector.role_scores(question_vectors, memory)
    gates = selector.gate_weights(question_vectors)
    fused = torch.zeros_like(role_scores[ROLE_NAMES[0]])
    for index, role in enumerate(ROLE_NAMES):
        fused = fused + gates[:, index:index + 1].to(fused.dtype) * role_scores[role]
    return role_scores, gates, fused


@torch.no_grad()
def role_mass_budget_sweep(
    selector: RoleAwareSelector,
    question_vectors: torch.Tensor,
    memory: LatentMemory,
    *,
    budget_ratios: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    temperature: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[float, torch.Tensor], torch.Tensor]:
    """Return online-predictable role-score mass at each discrete budget.

    The score uses no role labels.  It measures how much probability mass from
    each predicted evidence head is retained by that budget, then takes the
    minimum across key, answer and support.  The running maximum makes the
    stopping confidence monotone even when exact quota rounding causes two
    adjacent diversified selections not to be nested.
    """
    if question_vectors.ndim != 2 or question_vectors.shape[0] != 1:
        raise ValueError("dynamic role selection currently requires one query")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    ratios = sorted({float(value) for value in budget_ratios})
    if not ratios or any(not 0.0 < value <= 1.0 for value in ratios):
        raise ValueError("budget ratios must contain values in (0, 1]")
    role_scores, gates, fused = fused_role_scores(selector, question_vectors, memory)
    probabilities = {
        role: torch.softmax(role_scores[role][0].float() / temperature, dim=0)
        for role in PREDICTED_ROLES
    }
    selections: dict[float, torch.Tensor] = {}
    sweep: list[dict[str, Any]] = []
    monotone_confidence = 0.0
    for ratio in ratios:
        k = max(1, min(memory.num_latents, math.ceil(ratio * memory.num_latents)))
        indices = selector._diversified_indices(role_scores, fused, k, gates)[0]
        selections[ratio] = indices
        masses = {
            role: float(probabilities[role].index_select(0, indices).sum().item())
            for role in PREDICTED_ROLES
        }
        raw_confidence = min(masses.values())
        monotone_confidence = max(monotone_confidence, raw_confidence)
        sweep.append({
            "ratio": ratio,
            "k": int(indices.numel()),
            "predicted_coverage": monotone_confidence,
            "predicted_coverage_raw": raw_confidence,
            "predicted_role_mass": masses,
        })
    return sweep, selections, fused[0]


@torch.no_grad()
def select_calibrated_role_budget(
    selector: RoleAwareSelector,
    question_vectors: torch.Tensor,
    memory: LatentMemory,
    *,
    minimum_ratio: float = 0.25,
    budget_ratios: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    coverage_threshold: float,
    temperature: float,
    forced_ratio: float | None = None,
) -> DynamicSelectionOutput:
    """Choose the first calibrated role-mass budget and preserve source order."""
    if not 0.0 < minimum_ratio <= 1.0:
        raise ValueError("minimum ratio must be in (0, 1]")
    if not 0.0 <= coverage_threshold <= 1.0:
        raise ValueError("coverage threshold must be in [0, 1]")
    ratios = sorted({float(value) for value in budget_ratios if float(value) >= minimum_ratio})
    if not ratios:
        ratios = [1.0]
    sweep, selections, fused = role_mass_budget_sweep(
        selector,
        question_vectors,
        memory,
        budget_ratios=ratios,
        temperature=temperature,
    )
    if forced_ratio is not None:
        chosen = next(
            (point for point in sweep if abs(float(point["ratio"]) - forced_ratio) < 1e-12),
            None,
        )
        if chosen is None:
            raise ValueError("forced ratio is absent from the calibrated budget grid")
    else:
        chosen = next(
            (point for point in sweep if float(point["predicted_coverage"]) >= coverage_threshold),
            sweep[-1],
        )
    ratio = float(chosen["ratio"])
    indices = selections[ratio]
    return DynamicSelectionOutput(
        indices=indices,
        scores=fused,
        selected_count=int(indices.numel()),
        selected_ratio=ratio,
        predicted_coverage=float(chosen["predicted_coverage"]),
        budget_sweep=sweep,
        selected_coverage=float(chosen["predicted_coverage_raw"]),
    )
