"""Normalize raw DocETL extractions into the final output schema."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .url_builder import (
    build_dataset_url,
    infer_repository,
    normalize_identifier,
    normalize_repository,
)


def normalize_record(paper: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    """Convert a single (paper, raw extraction) pair into the final schema."""
    raw_id = item.get("dataset_identifier") or ""
    raw_repo = item.get("repository") or ""
    norm_id = normalize_identifier(raw_id)
    canonical_repo = infer_repository(norm_id, raw_repo) or normalize_repository(raw_repo)
    url = build_dataset_url(norm_id, canonical_repo)
    return {
        "paper_id": paper.get("paper_id", ""),
        "paper_doi": paper.get("paper_doi", ""),
        "pmcid": paper.get("pmcid", ""),
        "pmid": paper.get("pmid", ""),
        "dataset_identifier": norm_id,
        "repository": canonical_repo,
        "url": url,
        "evidence": (item.get("evidence") or "")[:500],
        "confidence": item.get("confidence") or "",
        "notes": item.get("notes") or "",
    }


def flatten_docetl_output(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten DocETL output rows that contain a list of dataset references.

    Each input row is expected to carry the source paper fields plus a
    `dataset_references` list. We emit one final row per dataset reference.
    Papers with no references emit a single placeholder row with empty
    identifier (useful for coverage analysis).
    """
    out: list[dict[str, Any]] = []
    for row in records:
        paper = {
            "paper_id": row.get("paper_id", ""),
            "paper_doi": row.get("paper_doi", ""),
            "pmcid": row.get("pmcid", ""),
            "pmid": row.get("pmid", ""),
        }
        refs = row.get("dataset_references") or []
        if not refs:
            out.append({
                **paper,
                "dataset_identifier": "",
                "repository": "",
                "url": "",
                "evidence": "",
                "confidence": "",
                "notes": "no dataset references found",
            })
            continue
        seen_ids: set[str] = set()
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            normalized = normalize_record(paper, ref)
            ds_id = normalized["dataset_identifier"]
            if not ds_id:
                # Skip empty identifiers from the LLM
                continue
            if ds_id in seen_ids:
                # Drop within-paper duplicates the LLM emitted twice
                continue
            seen_ids.add(ds_id)
            out.append(normalized)
    return out


def write_predictions(rows: list[dict[str, Any]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
