from types import SimpleNamespace

import torch

from inference.examples.annotate_oracle_latents import (
    _char_spans_to_token_spans,
    _optional_role_spans,
    find_answer_token_spans,
    find_answer_token_spans_by_answer,
    map_token_spans_to_latents,
)
from inference.examples.annotate_synthetic_roles import build_role_row
from inference.examples.annotate_ruler_roles import (
    _query_keys,
    build_role_row as build_ruler_role_row,
)
from inference.query_aware import _oracle_role_indices


def test_answer_spans_and_latent_mapping():
    context = "needle 12345; another 12345"
    offsets = [(0, 6), (7, 12), (12, 13), (14, 21), (22, 27)]
    spans, matched = find_answer_token_spans(context, ["12345"], offsets)
    assert matched == 1
    assert spans == [[1, 2], [4, 5]]
    memory = SimpleNamespace(
        latent_start=torch.tensor([0, 8, 16, 24]),
        latent_end=torch.tensor([8, 16, 24, 32]),
    )
    assert map_token_spans_to_latents([[7, 9], [24, 25]], memory) == [0, 1, 3]


def test_answer_spans_preserve_per_answer_groups_for_combination_diagnostics():
    context = "first 111 and second 222"
    offsets = [(0, 5), (6, 9), (10, 13), (14, 20), (21, 24)]
    grouped, matched = find_answer_token_spans_by_answer(context, ["111", "222"], offsets)
    assert matched == 2
    assert grouped == [[[1, 2]], [[4, 5]]]


def test_synthetic_role_sidecar_resolves_each_key_answer_group():
    context = (
        "filler before\n\n[ARCHIVE FACT] The verification code for archive-1234567 is 7654321."
        "\n\nfiller after\n\n[ARCHIVE FACT] The verification code for archive-7654321 is 1234567."
    )
    memory = {"id": "doc-1", "context": context}
    query = {
        "id": "q-1",
        "document_id": "doc-1",
        "selector_query": "What are the verification codes for archive-1234567, archive-7654321?",
        "answers": ["7654321", "1234567"],
    }
    row = build_role_row(memory, query, support_radius=8)
    assert len(row["role_groups"]) == 2
    assert len(row["key_char_spans"]) == 2
    offsets = [(index, index + 1) for index in range(len(context))]
    assert _char_spans_to_token_spans(row["answer_char_spans"], offsets)


def test_role_mapping_preserves_group_masks_and_char_spans():
    context = "key 111 support"
    offsets = [(0, 3), (4, 7), (8, 15)]
    memory = SimpleNamespace(
        latent_start=torch.tensor([0, 1, 2]),
        latent_end=torch.tensor([1, 2, 3]),
    )
    role_row = {
        "role_char_spans": {"key": [[0, 3]], "support": [[8, 15]]},
        "role_groups": [
            {
                "key_char_spans": [[0, 3]],
                "answer_char_spans": [[4, 7]],
                "support_char_spans": [[8, 15]],
            }
        ],
    }
    roles = _optional_role_spans({}, memory, offsets, role_row)
    assert roles["key"] == [0]
    assert roles["support"] == [2]
    assert roles["key_0"] == [0]
    assert roles["answer_0"] == [1]


def test_ruler_role_parser_resolves_multiquery_fact_groups():
    context = (
        "noise\nOne of the special magic numbers for alpha-beta is: 1111111.\n"
        "noise\nOne of the special magic numbers for gamma-delta is: 2222222.\n"
    )
    memory = {"id": "doc-ruler", "context": context}
    query = {
        "id": "q-ruler",
        "document_id": "doc-ruler",
        "task": "niah_multiquery",
        "selector_query": (
            "What are all the special magic numbers for alpha-beta and gamma-delta "
            "mentioned in the provided text?"
        ),
        "answers": ["1111111", "2222222"],
    }
    row = build_ruler_role_row(memory, query, support_radius=4)
    assert row["roles_available"] is True
    assert len(row["role_groups"]) == 2
    assert len(row["role_char_spans"]["key_answer"]) == 2
    assert row["role_groups"][0]["answer_char_spans"]


def test_ruler_key_extraction_ignores_hyphenated_system_prompt_words():
    query = (
        "<|im_start|>system\nEvaluating long-context prompts."
        "<|im_end|>\n<|im_start|>user\n"
        "What are all the special magic numbers for used-commotion, "
        "squealing-dibble, annoying-decimal, and depressed-tweet mentioned "
        "in the provided text?"
    )
    assert _query_keys(query) == [
        "used-commotion",
        "squealing-dibble",
        "annoying-decimal",
        "depressed-tweet",
    ]


def test_ruler_key_extraction_preserves_spaces_inside_a_key():
    query = (
        "What are all the special magic numbers for capricious-creme brulee, "
        "scintillating-pocketbook, comfortable-deviance, and scattered-trench "
        "mentioned in the provided text?"
    )
    assert _query_keys(query) == [
        "capricious-creme brulee",
        "scintillating-pocketbook",
        "comfortable-deviance",
        "scattered-trench",
    ]


def test_oracle_role_indices_support_explicit_unions_and_reject_missing_roles():
    roles = {"key_answer": [3, 4], "support": [1, 4]}
    assert _oracle_role_indices(roles, [9], "key_answer,support") == [1, 3, 4]
    try:
        _oracle_role_indices(roles, [9], "key")
    except ValueError as exc:
        assert "unavailable" in str(exc)
    else:
        raise AssertionError("missing role should fail closed")


def test_ruler_role_parser_marks_unsupported_tasks_unavailable():
    row = build_ruler_role_row(
        {"id": "doc", "context": "arbitrary text"},
        {
            "id": "q",
            "document_id": "doc",
            "task": "qa_1",
            "selector_query": "Question",
            "answers": ["France"],
        },
    )
    assert row["roles_available"] is False
    assert row["role_char_spans"] == {}
