import math

import pytest
import torch
from types import SimpleNamespace

from inference.examples.annotate_utility_oracle import build_candidate_family
from inference.query_aware import _greedy_decode_with_prefill
from latent_context.positioning import build_decoder_position_ids
from latent_context.utility import answer_nll_metrics, exact_group_oracle_candidates


def test_full_layout_positions_preserve_sparse_latent_and_suffix_locations():
    attention_mask = torch.ones(1, 10, dtype=torch.long)
    positions = build_decoder_position_ids(
        attention_mask,
        [[(2, 5)]],
        [torch.tensor([0, 2, 5])],
        [6],
        policy="full_layout",
    )
    assert positions.tolist() == [[0, 1, 2, 4, 7, 8, 9, 10, 11, 12]]
    compact = build_decoder_position_ids(
        attention_mask, [[(2, 5)]], [[0, 2, 5]], [6], policy="compact"
    )
    assert compact.tolist() == [list(range(10))]


def test_full_layout_positions_support_empty_memory_control():
    positions = build_decoder_position_ids(
        torch.ones(1, 5, dtype=torch.long), [[(2, 2)]], [[]], [4], policy="full_layout"
    )
    assert positions.tolist() == [[0, 1, 6, 7, 8]]
    with pytest.raises(ValueError, match="strictly source ordered"):
        build_decoder_position_ids(
            torch.ones(1, 5, dtype=torch.long), [[(1, 3)]], [[2, 1]], [4]
        )


def test_answer_nll_retention_exposes_regressions_and_undefined_denominator():
    metrics = answer_nll_metrics(full_nll=1.0, selected_nll=2.0, null_nll=3.0)
    assert metrics["answer_nll_delta_vs_full"] == 1.0
    assert metrics["answer_nll_retention"] == 0.5
    assert metrics["answer_nll_retention_clipped"] == 0.5
    assert math.isclose(metrics["answer_likelihood_ratio_vs_full"], math.exp(-1.0))

    worse = answer_nll_metrics(full_nll=1.0, selected_nll=4.0, null_nll=3.0)
    assert worse["answer_nll_retention"] == -0.5
    assert worse["answer_nll_retention_clipped"] == 0.0
    assert answer_nll_metrics(3.0, 2.0, 1.0)["answer_nll_retention"] is None


def test_group_utility_candidates_are_exact_budget_and_auditable():
    candidates = exact_group_oracle_candidates(459, selection_ratio=0.25, num_groups=8)
    assert candidates
    assert all(len(indices) == 115 for indices in candidates)
    assert all(tuple(sorted(indices)) == indices for indices in candidates)

    family = build_candidate_family(
        memory_length=16,
        k=4,
        selection_ratio=0.25,
        num_groups=8,
        contribution_values=list(range(16)),
        span_indices=[2],
        external_candidates={"learned": [0, 3, 9, 15]},
    )
    assert all(len(indices) == 4 for indices in family)
    names = {name for definitions in family.values() for name in definitions}
    assert {"uniform", "contribution_topk", "span_oracle_fill", "learned"} <= names


def test_utility_candidates_include_span_neighbor_and_provenance_fields():
    family = build_candidate_family(
        memory_length=16,
        k=4,
        selection_ratio=0.25,
        num_groups=8,
        span_indices=[5],
        neighbor_radius=1,
    )
    names = {name for definitions in family.values() for name in definitions}
    assert "span_oracle_neighbor_r1" in names


def test_utility_candidate_family_adds_key_answer_support_unions():
    family = build_candidate_family(
        memory_length=16,
        k=4,
        selection_ratio=0.25,
        num_groups=8,
        role_indices={"key": [1], "answer": [5], "support": [9]},
    )
    names = {name for definitions in family.values() for name in definitions}
    assert {"key_role_fill", "answer_role_fill", "support_role_fill", "key_answer_support_union"} <= names


def test_incremental_decode_continues_after_virtual_full_layout_position():
    class RecordingDecoder:
        def __init__(self):
            self.position_ids = []

        def __call__(self, input_ids=None, inputs_embeds=None, position_ids=None, **kwargs):
            self.position_ids.append(position_ids.clone())
            length = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
            logits = torch.zeros(1, length, 4)
            logits[..., 1] = 1
            return SimpleNamespace(logits=logits, past_key_values=object())

    decoder = RecordingDecoder()
    generated, _, _ = _greedy_decode_with_prefill(
        decoder,
        inputs_embeds=torch.zeros(1, 4, 3),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        position_ids=torch.tensor([[0, 3, 7, 9]]),
        max_tokens=3,
        eos_token_id=None,
        device="cpu",
    )
    assert generated.shape == (1, 3)
    assert [positions.tolist() for positions in decoder.position_ids] == [
        [[0, 3, 7, 9]], [[10]], [[11]],
    ]
