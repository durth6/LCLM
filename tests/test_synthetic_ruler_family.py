"""Unit tests for the synthetic cwe/fwe/vt/qa family generator."""

import random

import pytest

from improvement_round1.d3_nonniah_train.prepare_synthetic_ruler_family import (
    _build_cwe,
    _build_fwe,
    _build_qa,
    _build_vt,
    _insert_pieces,
    _role_row,
)

FILLER = [
    "The quiet harbour kept its records in careful order for many years. " * 40,
    "Old merchants traded grain and cloth across the narrow strait. " * 40,
    "A long dispute over taxes ended when the council met in autumn. " * 40,
]


def _built(builder, seed: int = 11):
    return builder(FILLER, random.Random(seed), 4000, seed)


def test_insert_pieces_records_every_duplicate_occurrence():
    context, spans = _insert_pieces(
        "0123456789", [(2, "ab"), (6, "ab"), (0, "ab")]
    )
    assert context.count("ab") == 3
    assert len(spans) == 3
    extracted = [context[start:end] for start, end in spans]
    assert extracted == ["ab", "ab", "ab"]
    assert spans == sorted(spans)


def test_cwe_builds_ten_repeated_common_words():
    built = _built(_build_cwe)
    assert len(built["answers"]) == 10
    assert built["question"] == "What are the 10 most common words in the above text?"
    assert not any(answer in built["question"] for answer in built["answers"])
    for answer in built["answers"]:
        assert built["context"].count(f"\n{answer}\n") >= 22
    assert len(built["groups"]) == 10
    for group in built["groups"]:
        assert len(group["answer_char_spans"]) >= 22
        start, end = group["answer_char_spans"][0]
        assert built["context"][start:end] in built["answers"]
        assert "key_char_spans" not in group


def test_fwe_answers_rank_highest_frequency():
    built = _built(_build_fwe, seed=13)
    assert built["question"] == "What are the three most frequent words in the above text?"
    assert len(built["answers"]) == 3
    counts = {
        word: built["context"].count(f"\n{word}\n")
        for word in set(built["answers"])
    }
    occurrences = [built["context"].count(f"\n{answer}\n") for answer in built["answers"]]
    # The three asked words must not be outranked inside the answer set order.
    assert occurrences == sorted(occurrences, reverse=True)
    assert len(built["groups"]) == 3
    for answer in built["answers"]:
        assert answer not in built["question"]


def test_vt_answers_resolve_to_chain_roots():
    built = _built(_build_vt, seed=17)
    assert built["question"].startswith("Find all the values assigned to the variables ")
    assert len(built["groups"]) == len(built["answers"])
    for answer, group in zip(built["answers"], built["groups"]):
        assert built["context"].count(answer) == 1  # value appears exactly once
        start, end = group["answer_char_spans"][0]
        assert built["context"][start:end] == answer
        key_start, key_end = group["key_char_spans"][0]
        assert built["context"][key_start:key_end]
        chain_start, chain_end = group["key_answer_char_spans"][0]
        assert chain_start < chain_end
        assert answer not in built["question"]


def test_qa_asks_one_attribute_without_leaking_the_answer():
    for seed in (19, 23):
        built = _build_qa(FILLER, random.Random(seed), 4000, seed)
        assert built["question"] in (
            f"When was the chronicle of {built['question'].split('chronicle of ')[1].split(' ')[0]} completed?",
            f"Where was the chronicle of {built['question'].split('chronicle of ')[1].split(' ')[0]} completed?",
        )
        assert len(built["groups"]) == len(built["answers"])
        for answer, group in zip(built["answers"], built["groups"]):
            assert answer not in built["question"]
            start, end = group["answer_char_spans"][0]
            assert built["context"][start:end] == answer
            sentence_start, sentence_end = group["key_answer_char_spans"][0]
            assert "chronicle" in built["context"][sentence_start:sentence_end]


def test_role_row_matches_oracle_sidecar_contract():
    built = _built(_build_cwe)
    row = _role_row("synthetic-ruler-family-000000", built)
    assert row["role_span_schema_version"] == 1
    assert row["id"] == row["document_id"]
    assert row["key_char_spans"] == []  # cwe has no key role
    assert len(row["answer_char_spans"]) >= 220
    for group in row["role_groups"]:
        assert group["answer_char_spans"]
        assert group["key_answer_char_spans"]
        assert group["support_char_spans"]
    vt_row = _role_row("synthetic-ruler-family-000001", _built(_build_vt))
    assert len(vt_row["key_char_spans"]) == len(vt_row["role_groups"])


def test_builder_rejects_missing_filler():
    with pytest.raises(Exception):
        _build_cwe([], random.Random(1), 4000, 1)
