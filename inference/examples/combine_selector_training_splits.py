"""Merge independently generated selector-training splits without losing provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine latent-only selector training splits")
    parser.add_argument("--input-dirs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    memories: list[dict] = []
    queries: list[dict] = []
    seen: set[str] = set()
    provenance = []
    for raw_dir in args.input_dirs:
        directory = Path(raw_dir)
        memory_rows = _read(directory / "memory.jsonl")
        query_rows = _read(directory / "queries.jsonl")
        memory_ids = {str(row["id"]) for row in memory_rows}
        query_ids = {str(row["id"]) for row in query_rows}
        if memory_ids != query_ids:
            raise ValueError(f"memory/query ids differ in {directory}")
        if seen & memory_ids:
            raise ValueError(f"duplicate ids across splits: {sorted(seen & memory_ids)[:3]}")
        if any("context" in row for row in query_rows):
            raise ValueError(f"online query contains raw context in {directory}")
        seen.update(memory_ids)
        memories.extend(memory_rows)
        queries.extend(query_rows)
        provenance.append(json.loads((directory / "manifest.json").read_text()))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "memory.jsonl", memories)
    _write(out / "queries.jsonl", queries)
    (out / "manifest.json").write_text(json.dumps({
        "split": "train",
        "purpose": "mixed-length Stage-A selector training only",
        "num_examples": len(memories),
        "components": provenance,
        "online_contract": "queries.jsonl contains no raw source context",
    }, indent=2) + "\n")
    print(json.dumps({"out_dir": str(out), "examples": len(memories), "components": len(provenance)}))


if __name__ == "__main__":
    main()
