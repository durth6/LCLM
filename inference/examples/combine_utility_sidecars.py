"""Merge sharded finite-candidate utility sidecars with provenance checks.

Utility rows are offline teacher-forced diagnostics.  This merger keeps the
same query order as the source JSONL and rejects mixed caches, schemas,
documents, or position policies before a sidecar can be used for training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


UTILITY_SCHEMA_VERSION = 2


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _normal_indices(value: Any, *, field: str, record_id: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{field} for {record_id} must be a list")
    try:
        indices = [int(index) for index in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} for {record_id} contains a non-integer") from exc
    if indices != sorted(set(indices)):
        raise ValueError(f"{field} for {record_id} must be source ordered and unique")
    if any(index < 0 for index in indices):
        raise ValueError(f"{field} for {record_id} contains a negative index")
    return indices


def _validate_row(row: dict[str, Any]) -> tuple[str, str, str | None, int, int]:
    if "id" not in row:
        raise ValueError("utility row is missing id")
    record_id = str(row["id"])
    document = row.get("document_id", row.get("cache_id", record_id))
    if not str(document):
        raise ValueError(f"utility row {record_id} has an empty document_id")
    schema = row.get("utility_oracle_schema_version")
    if schema is None or int(schema) != UTILITY_SCHEMA_VERSION:
        raise ValueError(
            f"utility row {record_id} has unsupported schema {schema!r}; "
            f"expected {UTILITY_SCHEMA_VERSION}"
        )
    position_policy = row.get("position_policy")
    if position_policy != "full_layout":
        raise ValueError(f"utility row {record_id} must use full_layout")
    fingerprint = row.get("cache_fingerprint")
    if not fingerprint:
        raise ValueError(f"utility row {record_id} is missing cache_fingerprint")
    candidates = row.get("candidate_answer_nll")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"utility row {record_id} has no candidate_answer_nll")
    expected_count = row.get("candidate_count")
    if expected_count is not None and int(expected_count) != len(candidates):
        raise ValueError(f"utility row {record_id} has an incorrect candidate_count")
    best_indices = _normal_indices(
        row.get("utility_oracle_latent_indices"),
        field="utility_oracle_latent_indices",
        record_id=record_id,
    )
    k_decoder = row.get("k_decoder")
    if k_decoder is not None and int(k_decoder) != len(best_indices):
        raise ValueError(f"utility row {record_id} has an inconsistent k_decoder")
    memory_length = int(row.get("full_latent_count", 0))
    if memory_length <= 0 or any(index >= memory_length for index in best_indices):
        raise ValueError(f"utility row {record_id} has out-of-range oracle indices")
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError(f"utility row {record_id} contains a malformed candidate")
        indices = _normal_indices(
            candidate.get("candidate_latent_indices"),
            field="candidate_latent_indices",
            record_id=record_id,
        )
        candidate_k = candidate.get("k_decoder")
        if candidate_k is not None and int(candidate_k) != len(indices):
            raise ValueError(f"utility row {record_id} has an inconsistent candidate k_decoder")
        if len(indices) != len(best_indices):
            raise ValueError(f"utility row {record_id} mixes candidate budgets")
        if any(index >= memory_length for index in indices):
            raise ValueError(f"utility row {record_id} has an out-of-range candidate index")
        if candidate.get("answer_nll") is None or candidate.get("candidate_gain") is None:
            raise ValueError(f"utility row {record_id} has an incomplete candidate score")
    return (
        record_id,
        str(document),
        str(fingerprint),
        int(row.get("cache_schema_version", 0)),
        int(schema),
    )


def merge_sidecars(
    inputs: list[str | Path],
    output: str | Path,
    queries_jsonl: str | Path | None = None,
) -> dict[str, Any]:
    """Merge utility shards, rejecting duplicate IDs and mixed provenance."""
    rows_by_id: dict[str, dict[str, Any]] = {}
    documents: dict[str, str] = {}
    fingerprints: set[str | None] = set()
    cache_schemas: set[int] = set()
    utility_schemas: set[int] = set()
    positions: set[str | None] = set()
    for raw_path in inputs:
        path = Path(raw_path)
        for row in _read(path):
            record_id, document, fingerprint, cache_schema, utility_schema = _validate_row(row)
            if record_id in rows_by_id:
                raise ValueError(f"duplicate utility target for {record_id}")
            rows_by_id[record_id] = row
            documents[record_id] = document
            fingerprints.add(fingerprint)
            cache_schemas.add(cache_schema)
            utility_schemas.add(utility_schema)
            positions.add(row.get("position_policy"))
    if not rows_by_id:
        raise ValueError("no utility rows found")
    if len(fingerprints) != 1 or len(cache_schemas) != 1 or len(utility_schemas) != 1 or len(positions) != 1:
        raise ValueError("utility shards have incompatible fingerprints, schemas, or position policies")

    if queries_jsonl is not None:
        queries = _read(Path(queries_jsonl))
        query_ids = [str(row["id"]) for row in queries]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("queries-jsonl contains duplicate IDs")
        missing = [record_id for record_id in query_ids if record_id not in rows_by_id]
        extra = [record_id for record_id in rows_by_id if record_id not in set(query_ids)]
        if missing or extra:
            raise ValueError(
                f"utility/query ID mismatch: missing={missing[:3]} extra={extra[:3]}"
            )
        for query in queries:
            record_id = str(query["id"])
            query_document = str(query.get("document_id", query.get("cache_id", record_id)))
            if query_document != documents[record_id]:
                raise ValueError(f"utility row {record_id} points to the wrong document")
        ordered_rows = [rows_by_id[record_id] for record_id in query_ids]
    else:
        ordered_rows = [rows_by_id[record_id] for record_id in sorted(rows_by_id)]

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for row in ordered_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "out": str(output_path),
        "records": len(ordered_rows),
        "utility_oracle_schema_version": next(iter(utility_schemas)),
        "cache_schema_version": next(iter(cache_schemas)),
        "cache_fingerprint": next(iter(fingerprints)),
        "position_policy": next(iter(positions)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge utility oracle sidecars")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--queries-jsonl", default=None)
    args = parser.parse_args()
    print(json.dumps(merge_sidecars(args.inputs, args.out, args.queries_jsonl), indent=2))


if __name__ == "__main__":
    main()
