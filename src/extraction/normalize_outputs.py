"""Normalize raw DocETL extractions into the final output schema.

Generic safety net layered on top of the LLM:

1. Schema validation. Every emitted identifier is checked against the regex
   declared for its inferred repository (config/repositories.yaml). Bare
   prefixes ("GSE"), shape-mismatched accessions ("19", "U55763" labelled
   as GEO/dbGaP, etc.), and similar are dropped.
2. Evidence grounding. The identifier must appear, case-insensitively, in
   the evidence string the LLM quoted from the paper. If the model "hallu-
   cinates" an identifier outside the quoted text, drop the row.
3. N/A safety valve. When the LLM emits a row with a blank identifier (i.e.,
   it found a dataset reference but no specific accession — "data available
   from X" with no ID) we preserve that as a row with identifier `N/A`
   rather than silently dropping it. When the LLM emits no rows at all, the
   paper still surfaces in the output with a single `N/A` row, so coverage
   metrics distinguish "found nothing" from "wasn't processed".
4. Within-paper deduplication on canonical (identifier, repository).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .registry import (
    NA_IDENTIFIER,
    accession_pattern,
    is_na_identifier,
    synonym_group,
    validate_identifier,
)
from .url_builder import (
    build_dataset_url,
    infer_repository,
    normalize_identifier,
    normalize_repository,
)


def _evidence_inconsistent(identifier: str, evidence: str) -> bool:
    """Return True only when the evidence positively contradicts the id.

    Three cases:
    - Empty evidence → cannot verify, accept (returns False).
    - Evidence contains the id (case- and whitespace-insensitive) → accept.
    - Evidence contains accession-like tokens, none of which match the id
      → reject (the model quoted one accession but emitted a different one,
      a strong hallucination signal).
    - Evidence is descriptive prose with no accession tokens → cannot
      verify, accept.

    Designed to flag clear hallucinations without punishing the LLM for
    quoting evidence that simply omitted the accession itself.
    """
    if not identifier or not evidence:
        return False
    norm_ev = "".join(evidence.split()).lower()
    norm_id = "".join(identifier.split()).lower()
    if norm_id in norm_ev:
        return False
    # Look for accession-like tokens in the evidence. If none, we can't
    # tell — be lenient.
    tokens = [t.lower() for t in accession_pattern().findall(evidence)]
    if not tokens:
        return False
    # The evidence quotes accession(s) but none match what we're emitting.
    return identifier.lower() not in tokens


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


def _na_row(paper: dict[str, Any], repository: str = NA_IDENTIFIER, evidence: str = "", notes: str = "no specific identifier extractable") -> dict[str, Any]:
    """Construct an N/A placeholder row for a paper."""
    return {
        "paper_id": paper.get("paper_id", ""),
        "paper_doi": paper.get("paper_doi", ""),
        "pmcid": paper.get("pmcid", ""),
        "pmid": paper.get("pmid", ""),
        "dataset_identifier": NA_IDENTIFIER,
        "repository": repository or NA_IDENTIFIER,
        "url": "",
        "evidence": (evidence or "")[:500],
        "confidence": "",
        "notes": notes,
    }


def flatten_docetl_output(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten DocETL output rows that contain a list of dataset references.

    Each input row is expected to carry the source paper fields plus a
    `dataset_references` list. Emits one final row per (validated) dataset
    reference. If the LLM produced no usable references for a paper, emits
    a single N/A placeholder so the paper still appears in coverage and
    can match an N/A groundtruth row.
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
        emitted_keys: set[tuple[str, str]] = set()
        emitted_any = False

        for ref in refs:
            if not isinstance(ref, dict):
                continue

            raw_id = ref.get("dataset_identifier") or ""
            raw_repo = ref.get("repository") or ""

            # Empty / N/A identifier from the LLM → preserve as a no-ID row.
            if is_na_identifier(raw_id):
                repo_label = normalize_repository(raw_repo) or (raw_repo.strip() if raw_repo else NA_IDENTIFIER)
                normalized = _na_row(
                    paper,
                    repository=repo_label,
                    evidence=ref.get("evidence") or "",
                    notes=ref.get("notes") or "no specific identifier in source",
                )
                key = (NA_IDENTIFIER.lower(), synonym_group(repo_label))
                if key in emitted_keys:
                    continue
                emitted_keys.add(key)
                out.append(normalized)
                emitted_any = True
                continue

            normalized = normalize_record(paper, ref)
            ds_id = normalized["dataset_identifier"]
            repo = normalized["repository"]

            # Schema check: drop bare prefixes ("GSE", "PXD"), shape-mismatched
            # accessions ("19" labelled DepMap, "U55763" labelled GEO).
            if repo and not validate_identifier(ds_id, repo):
                continue

            # Evidence consistency: drop only when the evidence quotes some
            # accession-like token but NOT the one we're emitting (a strong
            # hallucination signal). Lenient when evidence is descriptive
            # prose so we don't punish the LLM for under-quoting.
            evidence = normalized.get("evidence") or ""
            if _evidence_inconsistent(ds_id, evidence):
                continue

            # Dedup on synonym group, so PRIDE and ProteomeXchange rows for
            # the same PXD identifier collapse to one record.
            key = (ds_id.lower(), synonym_group(repo))
            if key in emitted_keys:
                continue
            emitted_keys.add(key)
            out.append(normalized)
            emitted_any = True

        if not emitted_any:
            # No usable rows — keep paper visible with a single N/A placeholder.
            out.append(_na_row(paper, notes="no dataset references found"))
    return out


def write_predictions(rows: list[dict[str, Any]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
