import re

import pytest
import torch

from latent_context.processor import LCLMProcessor


class TinyTokenizer:
    def __init__(self) -> None:
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


def test_selected_memory_processor_never_encodes_source_context():
    tokenizer = TinyTokenizer()
    processor = LCLMProcessor(tokenizer, tokenizer, compression_ratio=8)
    result = processor.process_selected_memory_batch(
        prompts=["Q:<|selected_memory|>"], selected_latent_counts=[3]
    )
    assert result["latent_counts"] == [[3]]
    assert result["memory_positions"] == [[(3, 6)]]
    assert result["input_ids"].shape[1] == 7


def test_selected_memory_processor_requires_exactly_one_marker():
    tokenizer = TinyTokenizer()
    processor = LCLMProcessor(tokenizer, tokenizer, compression_ratio=8)
    with pytest.raises(ValueError, match="exactly one"):
        processor.process_selected_memory_batch(["no marker"], [1])


def test_selected_memory_processor_can_build_explicit_empty_control():
    tokenizer = TinyTokenizer()
    processor = LCLMProcessor(tokenizer, tokenizer, compression_ratio=8)
    result = processor.process_selected_memory_batch(
        ["Q:<|selected_memory|>"], [0], allow_empty_memory=True
    )
    assert result["memory_positions"] == [[(3, 3)]]
    assert result["latent_counts"] == [[0]]
