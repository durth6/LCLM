"""Shared identifiers for latent-memory documents and their queries."""

from __future__ import annotations

from typing import Any, Mapping


def query_id(record: Mapping[str, Any]) -> str:
    """Return the unique query identifier required by sidecars and metrics."""
    value = record.get("id")
    if value is None or not str(value):
        raise ValueError("query records require a non-empty 'id'")
    return str(value)


def document_id(record: Mapping[str, Any]) -> str:
    """Return the cache/document identifier, with legacy one-query fallback."""
    value = record.get("document_id", record.get("cache_id", record.get("id")))
    if value is None or not str(value):
        raise ValueError("records require a non-empty document_id/cache_id/id")
    return str(value)
