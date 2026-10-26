import json

import pytest

from inference.examples.combine_utility_sidecars import merge_sidecars


def _row(record_id: str, document_id: str = "doc-1", fingerprint: str = "fp") -> dict:
    return {
        "id": record_id,
        "document_id": document_id,
        "utility_oracle_schema_version": 2,
        "position_policy": "full_layout",
        "cache_fingerprint": fingerprint,
        "cache_schema_version": 1,
        "full_latent_count": 8,
        "k_decoder": 2,
        "utility_oracle_latent_indices": [1, 6],
        "candidate_count": 1,
        "candidate_answer_nll": [{
            "candidate_latent_indices": [1, 6],
            "k_decoder": 2,
            "answer_nll": 0.2,
            "candidate_gain": 0.4,
        }],
    }


def _write(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_merge_utility_sidecars_checks_provenance_and_query_order(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    queries = tmp_path / "queries.jsonl"
    output = tmp_path / "merged.jsonl"
    _write(first, [_row("q-2")])
    _write(second, [_row("q-1")])
    _write(queries, [
        {"id": "q-1", "document_id": "doc-1"},
        {"id": "q-2", "document_id": "doc-1"},
    ])

    result = merge_sidecars([first, second], output, queries)
    assert result["records"] == 2
    assert [json.loads(line)["id"] for line in output.read_text().splitlines()] == ["q-1", "q-2"]


def test_merge_utility_sidecars_rejects_mixed_fingerprint(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write(first, [_row("q-1", fingerprint="fp-a")])
    _write(second, [_row("q-2", fingerprint="fp-b")])
    with pytest.raises(ValueError, match="incompatible fingerprints"):
        merge_sidecars([first, second], tmp_path / "out.jsonl")


def test_merge_utility_sidecars_rejects_unsorted_candidate(tmp_path):
    row = _row("q-1")
    row["candidate_answer_nll"][0]["candidate_latent_indices"] = [6, 1]
    path = tmp_path / "bad.jsonl"
    _write(path, [row])
    with pytest.raises(ValueError, match="source ordered"):
        merge_sidecars([path], tmp_path / "out.jsonl")
