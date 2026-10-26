import pytest
import torch

from inference.query_aware import _oracle_fill_indices, _requested_k


def test_requested_k_is_per_memory_ratio():
    assert _requested_k(453, None, 0.125) == 57
    assert _requested_k(553, None, 0.5) == 277
    assert _requested_k(10, 256, None) == 10
    with pytest.raises(ValueError):
        _requested_k(10, 2, 0.5)


def test_oracle_fill_preserves_oracle_and_budget():
    indices = _oracle_fill_indices([1, 7], memory_length=10, k=4, device="cpu")
    assert indices.numel() == 4
    assert {1, 7}.issubset(set(indices.tolist()))
    assert torch.all(indices[1:] > indices[:-1])
