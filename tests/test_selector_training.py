import torch
import pytest

from train.train_selector import (
    balanced_bce_loss,
    group_coverage_loss,
    multi_positive_ranking_loss,
    position_balance_loss,
    role_group_masks,
    role_coverage_loss,
    split_examples_by_document,
    utility_candidate_rank_loss,
)


def test_multi_positive_ranking_prefers_positive_latents():
    positives = torch.tensor([False, True, True, False])
    good = torch.tensor([-2.0, 3.0, 2.0, -2.0])
    bad = torch.tensor([3.0, -2.0, -2.0, 3.0])
    assert multi_positive_ranking_loss(good, positives) < multi_positive_ranking_loss(bad, positives)


def test_balanced_bce_handles_sparse_and_all_positive_labels():
    sparse = torch.tensor([False, True, False])
    assert balanced_bce_loss(torch.tensor([-2.0, 2.0, -2.0]), sparse) < 0.2
    assert balanced_bce_loss(torch.tensor([1.0, 2.0]), torch.tensor([True, True])).item() == 0.0


def test_role_coverage_loss_prefers_scores_covering_each_role():
    masks = {
        "answer": torch.tensor([True, False, False, False]),
        "support": torch.tensor([False, False, True, False]),
    }
    covered = role_coverage_loss(torch.tensor([2.0, -2.0, 2.0, -2.0]), masks)
    collapsed = role_coverage_loss(torch.tensor([4.0, 3.0, -3.0, -4.0]), masks)
    assert covered < collapsed


def test_group_coverage_loss_prefers_scores_covering_each_evidence_group():
    groups = {
        "key_answer_0": torch.tensor([True, False, False, False]),
        "key_answer_1": torch.tensor([False, False, True, False]),
    }
    covered = group_coverage_loss(
        torch.tensor([3.0, -3.0, 3.0, -3.0]), groups, selection_k=2
    )
    collapsed = group_coverage_loss(
        torch.tensor([5.0, 4.0, -5.0, -6.0]), groups, selection_k=2
    )
    assert covered < collapsed


def test_group_coverage_loss_ignores_empty_groups_and_validates_inputs():
    active = {"support_0": torch.tensor([False, True, False])}
    with_empty = {
        **active,
        "support_1": torch.tensor([False, False, False]),
    }
    scores = torch.tensor([-1.0, 2.0, -1.0])
    torch.testing.assert_close(
        group_coverage_loss(scores, active, selection_k=1),
        group_coverage_loss(scores, with_empty, selection_k=1),
    )
    with pytest.raises(ValueError, match="bool \[M\]"):
        group_coverage_loss(scores, {"bad": torch.ones(2, dtype=torch.bool)})
    with pytest.raises(ValueError, match="temperature"):
        group_coverage_loss(scores, active, temperature=0)
    with pytest.raises(ValueError, match="margin"):
        group_coverage_loss(scores, active, margin=-0.1)
    with pytest.raises(ValueError, match="selection_k"):
        group_coverage_loss(scores, active, selection_k=0)


def test_role_group_masks_require_exact_ascii_integer_suffixes():
    mask = torch.tensor([True, False])
    selected = role_group_masks(
        {
            "key_answer": mask,
            "key_answer_0": mask,
            "key_answer_01": mask,
            "key_answer_0_extra": mask,
            "key_answer_-1": mask,
            "key_answer_١": mask,
            "support_0": mask,
        },
        "key_answer",
    )
    assert set(selected) == {"key_answer_0", "key_answer_01"}


def test_utility_candidate_rank_loss_prefers_higher_gain_candidate():
    candidates = [
        {"latent_indices": [0, 1], "gain": 1.0},
        {"latent_indices": [2, 3], "gain": 0.0},
    ]
    good = utility_candidate_rank_loss(torch.tensor([2.0, 2.0, -2.0, -2.0]), candidates)
    bad = utility_candidate_rank_loss(torch.tensor([-2.0, -2.0, 2.0, 2.0]), candidates)
    assert good < bad
    assert position_balance_loss(torch.tensor([0.0, 0.0, 0.0, 0.0]), 2) == 0


def test_document_group_split_keeps_all_queries_for_a_document_together():
    examples = [
        {"id": f"q-{doc}-{query}", "document_id": doc}
        for doc in ("a", "b", "c", "d")
        for query in range(3)
    ]
    train, validation, manifest = split_examples_by_document(examples, 0.25, split_seed=11)
    train_docs = {row["document_id"] for row in train}
    validation_docs = {row["document_id"] for row in validation}
    assert train_docs.isdisjoint(validation_docs)
    assert manifest["document_overlap"] == []
    assert len(validation) == 3


def test_explicit_split_rejects_one_document_crossing_boundaries():
    examples = [
        {"id": "q1", "document_id": "doc", "split": "train"},
        {"id": "q2", "document_id": "doc", "split": "validation"},
    ]
    with pytest.raises(ValueError, match="multiple explicit splits"):
        split_examples_by_document(examples, 0.5, split_seed=1)
