import pytest

from inference.examples.prepare_ruler_niah_query_aware import FWE_PREAMBLE, split_lclm_prompt


def test_split_lclm_prompt_strips_context_from_online_artifacts():
    prompt = "prefix<|memory_start|>the hidden source<|memory_end|> question"
    context, selector_query, decoder_prompt = split_lclm_prompt(prompt)
    assert context == "the hidden source"
    assert "the hidden source" not in selector_query
    assert decoder_prompt == "prefix<|selected_memory|> question"


def test_split_lclm_prompt_requires_one_memory_region():
    with pytest.raises(ValueError, match="exactly one"):
        split_lclm_prompt("no memory")


def test_split_marker_free_fwe_prompt_uses_constrained_task_parser():
    prompt = f"prefix{FWE_PREAMBLE}aa bb aa\n\nQuestion: top words?"
    context, selector_query, decoder_prompt = split_lclm_prompt(prompt, task="fwe")
    assert context == "aa bb aa"
    assert "aa bb aa" not in selector_query
    assert decoder_prompt == f"prefix{FWE_PREAMBLE}<|selected_memory|>\n\nQuestion: top words?"


def test_split_moves_niah_trailing_question_outside_memory_region():
    prompt = (
        "prefix<|memory_start|>facts"
        "\nWhat are all the special magic numbers for item mentioned in the provided text? "
        "The special magic numbers for item mentioned in the provided text are\n"
        "<|memory_end|>suffix"
    )
    context, selector_query, decoder_prompt = split_lclm_prompt(prompt, task="niah_multivalue")
    assert context == "facts"
    assert "What are all the special magic numbers" in selector_query
    assert "facts" not in selector_query
    assert decoder_prompt == (
        "prefix<|selected_memory|>\nWhat are all the special magic numbers for item mentioned in the "
        "provided text? The special magic numbers for item mentioned in the provided text are\n"
        "suffix"
    )
