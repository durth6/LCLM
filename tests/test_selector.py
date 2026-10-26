from types import SimpleNamespace

import pytest
import torch

from latent_context.model import LCLM
from latent_context.selector import (
    BiEncoderSelector,
    CosineSelector,
    contiguous_coverage_groups,
    coverage_aware_greedy,
    coverage_for_selection,
    memory_structure_groups,
    LatentMemory,
    PositionConditioner,
    load_selector_checkpoint,
    save_selector_checkpoint,
    expand_with_neighbors,
    select_random_indices,
    select_topk_in_source_order,
    select_uniform_indices,
    select_dynamic_budget,
    source_span_summary,
)


def make_memory() -> LatentMemory:
    return LatentMemory(
        latents=torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 2.0, 0.0],
        ]),
        latent_start=torch.tensor([0, 8, 16, 24], dtype=torch.int32),
        latent_end=torch.tensor([8, 16, 24, 32], dtype=torch.int32),
        latent_index=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        segment_id=torch.zeros(4, dtype=torch.int16),
        window_id=torch.zeros(4, dtype=torch.int32),
        fingerprint="unit-test",
    )


def test_cosine_topk_restores_source_order():
    memory = make_memory()
    output = CosineSelector()(torch.tensor([[0.0, 1.0, 0.0]]), memory, k=2)
    assert output.indices.tolist() == [[1, 3]]
    assert output.selected_count.tolist() == [2]


def test_stable_ties_prefer_earlier_source_position():
    scores = torch.tensor([[1.0, 1.0, 0.5, 1.0]])
    output = select_topk_in_source_order(scores, k=2)
    assert output.indices.tolist() == [[0, 1]]


def test_selected_memory_tracks_gaps_and_requires_order():
    memory = make_memory()
    selected = memory.select(torch.tensor([0, 2, 3]))
    assert selected.latent_index.tolist() == [0, 2, 3]
    with pytest.raises(ValueError, match="source order"):
        memory.select(torch.tensor([3, 1]))


def test_position_conditioner_is_a_zero_init_identity():
    memory = make_memory().select(torch.tensor([0, 2, 3]))
    embeddings = torch.randn(3, 7)
    conditioned = PositionConditioner(decoder_dim=7, hidden_dim=5)(embeddings, memory, source_length=32)
    torch.testing.assert_close(conditioned, embeddings)


def test_biencoder_returns_source_ordered_indices():
    memory = make_memory()
    selector = BiEncoderSelector(input_dim=3, retrieval_dim=2)
    output = selector(torch.randn(2, 3), memory, k=3)
    assert output.indices.shape == (2, 3)
    assert torch.all(output.indices[:, 1:] > output.indices[:, :-1])


def test_biencoder_accepts_bf16_encoder_vectors_with_fp32_head():
    memory = make_memory()
    selector = BiEncoderSelector(input_dim=3, retrieval_dim=2)
    scores = selector.score(torch.randn(1, 3, dtype=torch.bfloat16), memory)
    assert scores.dtype == torch.float32


def test_memory_fingerprint_changes_when_encoder_artifact_changes():
    encoder = SimpleNamespace(
        compression_ratio=8,
        pooling="mean",
        encoder_window_size=1024,
        is_causal=True,
        boundary_overlap=0,
        embed_model=SimpleNamespace(config=SimpleNamespace(_name_or_path="base")),
    )
    first = SimpleNamespace(encoder=encoder, _memory_cache_identity="encoder-sha-a")
    second = SimpleNamespace(encoder=encoder, _memory_cache_identity="encoder-sha-b")
    assert LCLM.memory_fingerprint(first) != LCLM.memory_fingerprint(second)


def test_selector_checkpoint_round_trip(tmp_path):
    selector = BiEncoderSelector(input_dim=3, retrieval_dim=2)
    positioner = PositionConditioner(decoder_dim=5, hidden_dim=4)
    save_selector_checkpoint(tmp_path, selector, positioner, "encoder-fingerprint")
    loaded_selector, loaded_positioner, metadata = load_selector_checkpoint(
        tmp_path, "encoder-fingerprint"
    )
    assert metadata["selector_type"] == "bi_encoder"
    assert loaded_positioner is not None
    for first, second in zip(selector.parameters(), loaded_selector.parameters()):
        torch.testing.assert_close(first, second)


def test_uniform_and_random_baselines_are_unique_and_source_ordered():
    assert select_uniform_indices(10, 4).tolist() == [0, 3, 6, 9]
    generator = torch.Generator().manual_seed(7)
    random_indices = select_random_indices(10, 4, generator=generator)
    assert random_indices.numel() == 4
    assert torch.all(random_indices[1:] > random_indices[:-1])


def test_neighbor_expansion_is_source_ordered_and_respects_fixed_budget():
    # Highest-score seed 5 produces its complete local window; index 1 is the
    # next-ranked non-neighbour used to fill K=4 exactly.
    expanded = expand_with_neighbors(torch.tensor([5, 1, 2, 3, 0, 4, 6, 7]), 8, k=4, radius=1)
    assert expanded.tolist() == [1, 4, 5, 6]


def test_neighbor_expansion_handles_boundary_windows_without_duplicates():
    expanded = expand_with_neighbors(torch.tensor([0, 7, 1, 2, 3, 4, 5, 6]), 8, k=5, radius=2)
    assert expanded.tolist() == [0, 1, 2, 3, 7]


def test_coverage_greedy_preserves_exact_budget_and_covers_structural_groups():
    scores = torch.tensor([10.0, 9.0, 8.0, 7.0, 1.0, 0.0, -1.0, -2.0])
    groups = contiguous_coverage_groups(8, num_groups=4)
    selected = coverage_aware_greedy(scores, 4, groups, coverage_weight=0.5)
    assert selected.tolist() == sorted(selected.tolist())
    assert len(selected) == 4
    assert coverage_for_selection(selected, groups)["covered_group_count"] >= 3


def test_zero_coverage_weight_reduces_to_relevance_topk():
    scores = torch.tensor([10.0, 9.0, 8.0, 1.0, 0.0, -1.0])
    groups = contiguous_coverage_groups(6, num_groups=3)
    selected = coverage_aware_greedy(scores, 2, groups, coverage_weight=0.0)
    assert selected.tolist() == [0, 1]


def test_dynamic_budget_uses_discrete_ratios_and_source_order():
    scores = torch.arange(16, dtype=torch.float32)
    groups = contiguous_coverage_groups(16, num_groups=4)
    result = select_dynamic_budget(
        scores,
        minimum_ratio=0.25,
        budget_ratios=(0.25, 0.5, 0.75, 1.0),
        groups=groups,
        coverage_threshold=1.0,
        coverage_weight=0.5,
    )
    assert result.selected_ratio in {0.25, 0.5, 0.75, 1.0}
    assert result.selected_count == len(result.indices)
    assert result.indices.tolist() == sorted(result.indices.tolist())
    assert [point["ratio"] for point in result.budget_sweep] == [0.25, 0.5, 0.75, 1.0]


def test_dynamic_budget_uses_relevance_only_coverage_for_budget_decision():
    # The 25% relevance Top-K is concentrated in one source group.  A coverage
    # reranker can spread those two slots across two groups, but that cannot be
    # used to make the budget sweep claim 50% predicted coverage.
    scores = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0])
    groups = contiguous_coverage_groups(8, num_groups=4)
    result = select_dynamic_budget(
        scores,
        minimum_ratio=0.25,
        budget_ratios=(0.25, 0.5, 0.75, 1.0),
        groups=groups,
        coverage_threshold=0.5,
        coverage_weight=10.0,
    )
    assert result.budget_sweep[0]["predicted_coverage"] == 0.25
    assert result.selected_ratio == 0.5
    assert result.predicted_coverage == 0.5
    assert result.selected_coverage == 1.0


def test_source_span_summary_reports_extent_and_discontinuous_runs():
    summary = source_span_summary(make_memory(), torch.tensor([0, 1, 3]))
    assert summary == {
        "selected_source_start_min": 0,
        "selected_source_end_max": 32,
        "selected_source_span_mean": 8.0,
        "selected_source_token_extent": 32,
        "selected_source_run_count": 2,
    }
    with pytest.raises(ValueError, match="source order"):
        source_span_summary(make_memory(), [3, 1])
    with pytest.raises(ValueError, match="non-empty"):
        source_span_summary(make_memory(), [])


def test_memory_structure_groups_use_window_metadata_when_available():
    memory = make_memory()
    groups = memory_structure_groups(memory)
    assert len(groups) == 4
    assert list(groups.values()) == [(0,), (1,), (2,), (3,)]


def test_memory_structure_groups_preserve_prefix_for_single_window_fallback():
    memory = make_memory()
    groups = memory_structure_groups(memory, fallback_groups=2, prefix="segment")
    assert list(groups) == ["segment_00", "segment_01"]
    assert list(groups.values()) == [(0, 1), (2, 3)]
