import json
import re
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from inference.examples.annotate_contribution_latents import (
    _answer_only_loss_from_hidden,
    _replace_single_region,
    _teacher_target,
    annotate_query,
    build_teacher_target,
    compute_zgrad_contributions,
    leave_one_out_nll_deltas,
    pearson_correlation,
    rank_contributions,
    spearman_correlation,
)
from inference.examples.combine_contribution_sidecars import merge_sidecars
from latent_context.adapter import Adapter
from latent_context.processor import LCLMProcessor
from latent_context.selector import LatentMemory
from train.train_selector import (
    _build_examples,
    contribution_distillation_loss,
    contribution_target_distribution,
)


class TinyTokenizer:
    def __init__(self):
        self.ids = {
            "<|memory|>": 1,
            "<|memory_start|>": 2,
            "<|memory_end|>": 3,
        }
        self.pad_token_id = 0

    def convert_tokens_to_ids(self, token):
        return self.ids[token]

    def encode(self, text, add_special_tokens=False):
        pieces = re.findall(r"<\|[^|]+\|>|.", text)
        return [self.ids.get(piece, 10) for piece in pieces]

    def __call__(self, texts, padding, truncation, return_tensors):
        encoded = [self.encode(text) for text in texts]
        width = max(map(len, encoded))
        ids = [row + [self.pad_token_id] * (width - len(row)) for row in encoded]
        mask = [[1] * len(row) + [0] * (width - len(row)) for row in encoded]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


class TinyBackbone(nn.Module):
    def __init__(self, hidden=6, vocab=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.projection = nn.Linear(hidden, hidden)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        hidden = inputs_embeds if inputs_embeds is not None else self.embedding(input_ids)
        return SimpleNamespace(last_hidden_state=self.projection(hidden))


class TinyDecoder(nn.Module):
    def __init__(self, hidden=6, vocab=32):
        super().__init__()
        self.model = TinyBackbone(hidden, vocab)
        self.lm_head = nn.Linear(hidden, vocab)

    def get_input_embeddings(self):
        return self.model.embedding

    def get_output_embeddings(self):
        return self.lm_head


def make_memory(num_latents=3, dim=6):
    return LatentMemory(
        latents=torch.arange(num_latents * dim, dtype=torch.float32).reshape(num_latents, dim) / 10,
        latent_start=torch.arange(num_latents, dtype=torch.int32) * 8,
        latent_end=torch.arange(1, num_latents + 1, dtype=torch.int32) * 8,
        latent_index=torch.arange(num_latents, dtype=torch.int32),
        segment_id=torch.zeros(num_latents, dtype=torch.int16),
        window_id=torch.arange(num_latents, dtype=torch.int32),
        fingerprint="tiny",
    )


def test_teacher_target_and_zgrad_ranking_are_deterministic():
    assert build_teacher_target([123, "abc"]) == "123, abc"
    latents = torch.tensor([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
    gradients = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.1, 0.1]])
    contributions = compute_zgrad_contributions(latents, gradients)
    torch.testing.assert_close(contributions, torch.tensor([2.0, 3.0, 0.14142136]))
    assert rank_contributions(contributions).tolist() == [2, 1, 3]
    assert _teacher_target({
        "id": "qa",
        "answers": ["first", "second"],
        "scoring_function": "ruler_string_match_part",
    }, ", ") == "first"


def test_answer_only_loss_and_latent_injection_keep_gradient():
    hidden = torch.randn(1, 4, 5, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 3]])
    lm_head = nn.Linear(5, 8)
    loss, count = _answer_only_loss_from_hidden(hidden, labels, lm_head)
    assert count == 2
    loss.backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()

    text = torch.randn(1, 6, 5)
    projected = torch.randn(2, 5, requires_grad=True)
    combined = _replace_single_region(text, projected, [[(2, 4)]], expected_count=2)
    combined.square().sum().backward()
    assert projected.grad is not None
    assert torch.isfinite(projected.grad).all()


def test_contribution_distillation_prefers_matching_scores():
    contributions = torch.tensor([0.1, 2.0, 0.5, 0.0])
    target = contribution_target_distribution(contributions, temperature=0.5, normalization="minmax")
    assert target.shape == contributions.shape
    torch.testing.assert_close(target.sum(), torch.tensor(1.0))
    good = torch.log(target.clamp_min(1e-6))
    bad = -good
    assert contribution_distillation_loss(good, contributions, 0.5, "minmax") < contribution_distillation_loss(
        bad, contributions, 0.5, "minmax"
    )
    with pytest.raises(ValueError, match="non-negative"):
        contribution_target_distribution(torch.tensor([1.0, -1.0]))
    assert pearson_correlation([1, 2, 3], [2, 4, 6]) > 0.99
    assert spearman_correlation([3, 1, 2], [6, 2, 4]) > 0.99


def test_annotate_query_tiny_decoder_end_to_end():
    tokenizer = TinyTokenizer()
    processor = LCLMProcessor(tokenizer, tokenizer, compression_ratio=8)
    decoder = TinyDecoder()
    model = SimpleNamespace(adapter=Adapter(6, 6, adapter_type="mlp"), decoder=decoder)
    query = {
        "id": "tiny-0",
        "task": "unit",
        "prompt": "Q:<|selected_memory|>",
        "answers": ["42"],
    }
    row = annotate_query(model, processor, make_memory(), query, "cpu", activation_checkpointing=True)
    assert row["target_definition"] == "zgrad_pre_adapter_abs_l2_full_layout_v2"
    assert row["position_policy"] == "full_layout"
    assert len(row["latent_contributions"]) == 3
    assert len(row["latent_contribution_ranks"]) == 3
    assert sorted(row["latent_contribution_ranks"]) == [1, 2, 3]
    assert all(value >= 0 for value in row["latent_contributions"])
    assert row["teacher_answer_token_count"] > 0
    assert row["cache_fingerprint"] == "tiny"

    indices, deltas, full_nll = leave_one_out_nll_deltas(
        model, processor, make_memory(), query, "cpu", indices=[0, 2]
    )
    assert indices == [0, 2]
    assert len(deltas) == 2
    assert all(torch.isfinite(torch.tensor(delta)) for delta in deltas)
    assert full_nll >= 0


def test_contribution_sidecar_joins_oracle_examples_and_respects_limit(tmp_path):
    queries = tmp_path / "queries.jsonl"
    oracle = tmp_path / "oracle.jsonl"
    contribution = tmp_path / "contribution.jsonl"
    queries.write_text(
        '{"id":"a","selector_query":"q-a"}\n'
        '{"id":"b","selector_query":"q-b"}\n'
        '{"id":"c","selector_query":"q-c"}\n'
    )
    oracle.write_text(
        '{"id":"a","oracle_latent_indices":[0]}\n'
        '{"id":"b","oracle_latent_indices":[1]}\n'
        '{"id":"c","oracle_latent_indices":[2]}\n'
    )
    contribution.write_text(
        '{"id":"a","latent_contributions":[1,0]}\n'
        '{"id":"b","latent_contributions":[0,1]}\n'
        '{"id":"c","latent_contributions":[1,1]}\n'
    )
    examples = _build_examples(queries, oracle, contribution, max_examples=2)
    assert [item["id"] for item in examples] == ["a", "b"]
    assert examples[0]["contribution"]["values"] == [1.0, 0.0]


def test_merge_contribution_sidecars_checks_ids_and_preserves_query_order(tmp_path):
    shard_a = tmp_path / "a.jsonl"
    shard_b = tmp_path / "b.jsonl"
    queries = tmp_path / "queries.jsonl"
    out = tmp_path / "merged.jsonl"
    common = {
        "target_definition": "zgrad_pre_adapter_abs_l2_full_layout_v2",
        "cache_fingerprint": "tiny",
        "contribution_schema_version": 2,
        "position_policy": "full_layout",
    }
    shard_a.write_text(json.dumps({"id": "a", "latent_contributions": [1], **common}) + "\n")
    shard_b.write_text(json.dumps({"id": "b", "latent_contributions": [2], **common}) + "\n")
    queries.write_text('{"id":"b"}\n{"id":"a"}\n')
    result = merge_sidecars([shard_a, shard_b], out, queries)
    assert result["records"] == 2
    assert [json.loads(line)["id"] for line in out.read_text().splitlines()] == ["b", "a"]
