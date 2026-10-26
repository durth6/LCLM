"""Unit tests for multi-slot / sub-query selection policies."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from latent_context.query_encoder import QuestionEncoder
from latent_context.selector import (
    BiEncoderSelector,
    LatentMemory,
    content_token_slots,
    extract_entity_slots,
    reciprocal_rank_fusion,
    select_slots_union,
    select_topk_in_source_order,
    split_question_sentences,
    user_question_text,
)
from inference.query_aware import answer_queries


CHAT_TEMPLATE = (
    "<|im_start|>system\nYou are a precise assistant.<|im_end|>\n"
    "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
)

MULTIQUERY_QUESTION = (
    "Some special magic numbers are hidden within the following text.\n\n"
    "What are all the special magic numbers for used-commotion, squealing-dibble, "
    "annoying-decimal, and depressed-tweet mentioned in the provided text?"
)
MULTIKEY_QUESTION = (
    "A special magic number is hidden within the following text.\n\n"
    "What is the special magic number for innocent-clause mentioned in the provided text?"
)


def _memory(num_latents: int = 10, dim: int = 8, seed: int = 0) -> LatentMemory:
    generator = torch.Generator().manual_seed(seed)
    index = torch.arange(num_latents)
    return LatentMemory(
        latents=torch.randn(num_latents, dim, generator=generator),
        latent_start=index * 8,
        latent_end=index * 8 + 8,
        latent_index=index,
        segment_id=torch.zeros(num_latents, dtype=torch.long),
        window_id=torch.zeros(num_latents, dtype=torch.long),
        fingerprint="test-fingerprint",
    )


def test_user_question_text_strips_chat_template():
    query = CHAT_TEMPLATE.format(question=MULTIQUERY_QUESTION)
    assert user_question_text(query) == MULTIQUERY_QUESTION
    assert user_question_text(MULTIQUERY_QUESTION) == MULTIQUERY_QUESTION
    with pytest.raises(ValueError):
        user_question_text("   ")


def test_extract_entity_slots_decomposes_multi_entity_question():
    query = CHAT_TEMPLATE.format(question=MULTIQUERY_QUESTION)
    slots = extract_entity_slots(query)
    assert len(slots) == 5
    assert slots[0] == query
    for slot, entity in zip(slots[1:], [
        "used-commotion", "squealing-dibble", "annoying-decimal", "depressed-tweet",
    ]):
        assert f"for {entity} mentioned in the provided text?" in slot
        assert "," not in slot.split("for ")[-1].split(" mentioned")[0]


def test_extract_entity_slots_keeps_single_entity_question_intact():
    query = CHAT_TEMPLATE.format(question=MULTIKEY_QUESTION)
    assert extract_entity_slots(query) == [query]
    assert extract_entity_slots("plain question without enumeration?") == [
        "plain question without enumeration?"
    ]


def test_split_question_sentences_caps_to_last_sentences():
    query = CHAT_TEMPLATE.format(question=MULTIQUERY_QUESTION)
    slots = split_question_sentences(query, 2)
    assert slots[0] == query
    assert len(slots) == 3
    assert slots[-1].startswith("What are all the special magic numbers")
    with pytest.raises(ValueError):
        split_question_sentences(query, 0)


def test_select_slots_union_single_slot_matches_plain_topk():
    scores = torch.tensor([[0.1, 0.9, 0.3, 0.7, 0.5]])
    assert torch.equal(select_slots_union(scores, 3), torch.tensor([1, 3, 4]))
    reference = select_topk_in_source_order(scores, 3)
    assert torch.equal(select_slots_union(scores, 3), reference.indices[0])


def test_select_slots_union_exact_k_source_order_and_fill():
    # Three disjoint slots claim exactly their own latents; K equals the union.
    scores = torch.zeros(3, 10)
    scores[0, 0] = 5.0
    scores[0, 1] = 4.0
    scores[1, 5] = 5.0
    scores[1, 6] = 4.0
    scores[2, 8] = 5.0
    scores[2, 9] = 4.0
    selected = select_slots_union(scores, 6)
    assert torch.equal(selected, torch.tensor([0, 1, 5, 6, 8, 9]))


def test_select_slots_union_prunes_by_consensus_mean():
    # K=3, G=2 -> per-slot budget 2, union of four latents must shrink to three
    # by mean score, keeping the latent both slots agree on.
    scores = torch.zeros(2, 10)
    scores[0, 2] = 1.0
    scores[0, 0] = 0.2
    scores[0, 1] = 0.1
    scores[1, 3] = 1.0
    scores[1, 1] = 0.9
    scores[1, 0] = 0.0
    selected = select_slots_union(scores, 3)
    assert selected.numel() == 3
    assert 2 in selected.tolist() and 1 in selected.tolist()
    assert 0 not in selected.tolist()  # lowest mean score inside the union
    assert torch.equal(selected, torch.sort(selected).values)


def test_select_slots_union_handles_bounds():
    scores = torch.randn(4, 6, generator=torch.Generator().manual_seed(3))
    selected = select_slots_union(scores, 100)
    assert selected.numel() == 6  # clamped to M, exactly-K still holds
    assert torch.equal(selected, torch.sort(selected).values)
    with pytest.raises(ValueError):
        select_slots_union(torch.tensor([[1.0, float("nan")]]), 1)
    with pytest.raises(ValueError):
        select_slots_union(torch.zeros(0, 4), 1)


def test_reciprocal_rank_fusion_prefers_consistent_ranking():
    # Latent 1 ranks second on both slots; latent 0 wins one slot but is
    # irrelevant to the other, so RRF must prefer latent 1.
    scores = torch.tensor([
        [3.0, 2.9, 0.1, 0.0],
        [0.5, 2.9, 2.8, 0.4],
    ])
    fused = reciprocal_rank_fusion(scores)
    assert int(torch.argmax(fused)) == 1
    assert fused[0] > fused[2]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(scores, fusion_k=0)


def test_content_token_slots_filters_stopwords_and_caps():
    hidden = torch.randn(9, 4, generator=torch.Generator().manual_seed(5))
    mask = torch.ones(9, dtype=torch.long)
    texts = [",", "the", "magic", "innocent-clause", "42", "a", "number", "quiet", "zebra"]
    slots = content_token_slots(hidden, mask, texts, 16)
    assert slots.shape[0] == 6  # punctuation, stopwords and single letters removed
    slots = content_token_slots(hidden, mask, texts, 2)
    assert slots.shape[0] == 2
    # All-masked input falls back to a single slot instead of failing.
    assert content_token_slots(hidden, torch.zeros(9, dtype=torch.long), texts, 4).shape[0] == 1


class _StubBackbone(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        batch, query_length = input_ids.shape
        base = torch.arange(batch * query_length, dtype=torch.float32).reshape(batch, query_length, 1)
        hidden = base.repeat(1, 1, self.hidden_size) + self.dummy
        return SimpleNamespace(last_hidden_state=hidden)


def test_question_encoder_forward_tokens_returns_token_states():
    encoder = QuestionEncoder(_StubBackbone(4), hidden_size=4, freeze_backbone=True)
    input_ids = torch.tensor([[1, 2, 3, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    hidden, mask = encoder.forward_tokens(input_ids, attention_mask)
    assert hidden.shape == (1, 4, 4)
    assert torch.equal(mask, attention_mask)
    pooled = encoder(input_ids, attention_mask)
    expected = hidden[0, :3].mean(dim=0, keepdim=True)
    assert torch.allclose(pooled, expected, atol=1e-6)
    assert not encoder.backbone.dummy.requires_grad


def test_bi_encoder_score_slots_matches_pooled_score():
    selector = BiEncoderSelector(input_dim=8, retrieval_dim=4)
    memory = _memory()
    question = torch.randn(1, 8, generator=torch.Generator().manual_seed(7))
    assert torch.allclose(selector.score(question, memory), selector.score_slots(question, memory))
    slots = torch.randn(3, 8, generator=torch.Generator().manual_seed(8))
    slot_scores = selector.score_slots(slots, memory)
    assert slot_scores.shape == (3, memory.num_latents)
    assert torch.allclose(
        selector.score_slots(slots[:1], memory), slot_scores[:1].unsqueeze(0).squeeze(0).unsqueeze(0)
    )


def test_answer_queries_rejects_invalid_slot_configuration():
    with pytest.raises(ValueError):
        answer_queries("checkpoint", "cache", "queries", "out", None, 0.25, 128,
                       "cpu", "bf16", slot_policy="bogus")
    with pytest.raises(ValueError):
        answer_queries("checkpoint", "cache", "queries", "out", None, 0.25, 128,
                       "cpu", "bf16", slot_policy="entity", dynamic_k=True)
    with pytest.raises(ValueError):
        answer_queries("checkpoint", "cache", "queries", "out", None, 0.25, 128,
                       "cpu", "bf16", slot_combine="median")
