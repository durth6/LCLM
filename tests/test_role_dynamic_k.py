import torch

from latent_context.role_dynamic_k import (
    role_mass_budget_sweep,
    select_calibrated_role_budget,
)
from latent_context.role_selector import ROLE_NAMES
from latent_context.selector import LatentMemory


class _Selector:
    def role_scores(self, question, memory):
        base = torch.tensor([[8.0, 7.0, 1.0, 0.0]], device=question.device)
        return {role: base.clone() for role in ROLE_NAMES}

    def gate_weights(self, question):
        return torch.full((1, len(ROLE_NAMES)), 1.0 / len(ROLE_NAMES), device=question.device)

    def _diversified_indices(self, role_scores, fused, k, gates):
        ranked = torch.argsort(fused[0], descending=True, stable=True)[:k]
        return torch.sort(ranked).values.unsqueeze(0)


def _memory():
    values = torch.arange(4)
    return LatentMemory(
        latents=torch.zeros(4, 3),
        latent_start=values,
        latent_end=values + 1,
        latent_index=values,
        segment_id=torch.zeros(4, dtype=torch.long),
        window_id=torch.zeros(4, dtype=torch.long),
        fingerprint="test",
    )


def test_role_mass_dynamic_budget_uses_discrete_point_and_source_order():
    selector = _Selector()
    question = torch.zeros(1, 3)
    sweep, _, _ = role_mass_budget_sweep(
        selector, question, _memory(), budget_ratios=(0.25, 0.5, 1.0), temperature=1.0
    )
    assert [point["ratio"] for point in sweep] == [0.25, 0.5, 1.0]
    assert sweep[0]["predicted_coverage"] < sweep[1]["predicted_coverage"]
    result = select_calibrated_role_budget(
        selector,
        question,
        _memory(),
        minimum_ratio=0.25,
        budget_ratios=(0.25, 0.5, 1.0),
        coverage_threshold=float(sweep[1]["predicted_coverage"]),
        temperature=1.0,
    )
    assert result.selected_ratio == 0.5
    assert result.indices.tolist() == [0, 1]
    assert result.selected_count == 2


def test_unqualified_policy_can_force_fixed_50_fallback():
    result = select_calibrated_role_budget(
        _Selector(),
        torch.zeros(1, 3),
        _memory(),
        minimum_ratio=0.25,
        budget_ratios=(0.25, 0.5, 1.0),
        coverage_threshold=0.0,
        temperature=1.0,
        forced_ratio=0.5,
    )
    assert result.selected_ratio == 0.5
    assert result.indices.tolist() == [0, 1]
