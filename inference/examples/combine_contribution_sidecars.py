"""Merge sharded contribution sidecars with strict provenance checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def merge_sidecars(
    inputs: list[str | Path],
    output: str | Path,
    queries_jsonl: str | Path | None = None,
) -> dict[str, Any]:
    """Merge sidecars, rejecting duplicate IDs and incompatible targets."""
    rows_by_id: dict[str, dict[str, Any]] = {}
    definitions: set[Any] = set()
    fingerprints: set[Any] = set()
    schema_versions: set[Any] = set()
    position_policies: set[Any] = set()
    for raw_path in inputs:
        path = Path(raw_path)
        for row in _read(path):
            record_id = str(row["id"])
            if record_id in rows_by_id:
                raise ValueError(f"duplicate contribution target for {record_id}")
            values = row.get("latent_contributions")
            if not isinstance(values, list) or not values:
                raise ValueError(f"empty contribution target for {record_id}")
            definitions.add(row.get("target_definition"))
            fingerprints.add(row.get("cache_fingerprint"))
            schema_versions.add(row.get("contribution_schema_version"))
            position_policies.add(row.get("position_policy"))
            rows_by_id[record_id] = row

    if not rows_by_id:
        raise ValueError("no contribution rows found")
    if (
        len(definitions) != 1
        or len(fingerprints) != 1
        or len(schema_versions) != 1
        or len(position_policies) != 1
    ):
        raise ValueError(
            "sidecars have incompatible target definitions, fingerprints, schemas, or position policies"
        )

    if queries_jsonl is not None:
        queries = _read(Path(queries_jsonl))
        query_ids = [str(row["id"]) for row in queries]
        missing = [record_id for record_id in query_ids if record_id not in rows_by_id]
        extra = [record_id for record_id in rows_by_id if record_id not in set(query_ids)]
        if missing or extra:
            raise ValueError(
                f"sidecar/query ID mismatch: missing={missing[:3]} extra={extra[:3]}"
            )
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
        "target_definition": next(iter(definitions)),
        "cache_fingerprint": next(iter(fingerprints)),
        "contribution_schema_version": next(iter(schema_versions)),
        "position_policy": next(iter(position_policies)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sharded contribution sidecars")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--queries-jsonl", default=None,
                        help="optional query order and exact ID completeness check")
    args = parser.parse_args()
    print(json.dumps(merge_sidecars(args.inputs, args.out, args.queries_jsonl), indent=2))


if __name__ == "__main__":
    main()
