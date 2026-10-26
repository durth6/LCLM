"""Latent Context Language Models (LCLM).

An LCLM wraps an encoder LM (e.g. Qwen3-Embedding-0.6B), an MLP adapter,
and a decoder LM (e.g. Qwen3-4B-Instruct-2507) to compress a long input
sequence into a short sequence of latent tokens that the decoder consumes
in place of the original.

Public API
----------
``LCLM``         the full model (encoder + adapter + decoder)
``Encoder``      encoder + pooling stage
``Adapter``      MLP projection from encoder hidden dim to decoder hidden dim
``LCLMProcessor`` tokenizer + memory-span extractor for prompts
``from_pretrained(path)`` loads an LCLM checkpoint directory or HF repo
"""

from .adapter import Adapter
from .encoder import Encoder
from .model import LCLM
from .processor import LCLMProcessor
from .positioning import (
    DEFAULT_POSITION_POLICY, POSITION_POLICIES, build_decoder_position_ids,
)
from .query_encoder import QuestionEncoder
from .records import document_id, query_id
from .utility import answer_nll_metrics, exact_group_oracle_candidates
from .selector import (
    BiEncoderSelector, CosineSelector, DynamicSelectionOutput, LatentMemory, PositionConditioner,
    contiguous_coverage_groups, coverage_aware_greedy, coverage_for_selection,
    memory_structure_groups, source_span_summary,
    load_selector_checkpoint, save_selector_checkpoint, select_random_indices,
    select_dynamic_budget, select_uniform_indices,
)


def from_pretrained(checkpoint_path, **kwargs):
    """Load an LCLM checkpoint. Returns ``(model, decoder_tokenizer, processor)``."""
    from inference.hf import load_model
    return load_model(checkpoint_path, **kwargs)


def _lclm_from_pretrained(cls, path, **kwargs):
    """``LCLM.from_pretrained(path)`` — HF-style single-return convenience."""
    model, _, _ = from_pretrained(path, **kwargs)
    return model


LCLM.from_pretrained = classmethod(_lclm_from_pretrained)


__all__ = [
    "LCLM", "Encoder", "Adapter", "LCLMProcessor", "QuestionEncoder",
    "LatentMemory", "CosineSelector", "BiEncoderSelector", "PositionConditioner",
    "DynamicSelectionOutput", "contiguous_coverage_groups", "coverage_aware_greedy",
    "coverage_for_selection", "memory_structure_groups", "select_dynamic_budget",
    "source_span_summary",
    "load_selector_checkpoint", "save_selector_checkpoint", "select_random_indices",
    "select_uniform_indices", "DEFAULT_POSITION_POLICY", "POSITION_POLICIES",
    "build_decoder_position_ids", "document_id", "query_id", "from_pretrained",
    "answer_nll_metrics", "exact_group_oracle_candidates",
]
