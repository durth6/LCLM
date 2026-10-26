import random

from inference.examples.prepare_selector_synthetic_niah import (
    SELECTED_MEMORY,
    _chat_prompt,
    _make_context,
    _partition_source_documents,
)


def test_synthetic_selector_item_keeps_answers_out_of_the_online_query():
    context, question, answers = _make_context(
        ["A filler paragraph about geography and history." * 20],
        random.Random(11), target_chars=300, sample_index=0,
    )
    assert all(answer in context for answer in answers)
    assert all(answer not in question for answer in answers)
    prompt = _chat_prompt(question)
    assert prompt.count(SELECTED_MEMORY) == 1
    assert context not in prompt


def test_source_document_partition_is_disjoint_and_deterministic():
    documents = [(f"doc-{index}", f"text-{index}") for index in range(16)]
    first_train, first_validation = _partition_source_documents(documents, 0.25, 7)
    second_train, second_validation = _partition_source_documents(documents, 0.25, 7)
    assert first_train == second_train
    assert first_validation == second_validation
    assert {row[0] for row in first_train}.isdisjoint({row[0] for row in first_validation})


def test_hard_multikey_item_uses_requested_number_of_distinct_pairs():
    context, question, answers = _make_context(
        ["A filler paragraph about geography and history." * 100],
        random.Random(19),
        target_chars=2_000,
        sample_index=0,
        pair_count=12,
        asked_count=8,
    )
    assert context.count("[ARCHIVE FACT]") == 12
    assert len(answers) == len(set(answers)) == 8
    assert question.count("archive-") == 8
    assert all(answer in context and answer not in question for answer in answers)
