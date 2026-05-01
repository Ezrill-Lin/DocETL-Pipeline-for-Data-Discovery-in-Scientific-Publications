"""Load benchmark ground truth into a uniform schema.

Supported input formats:
- CSV / TSV with one row per (paper, dataset_reference)
- JSON / JSONL with the same shape

The DataRef-EXP / DataRef-REV files released with the DataGatherer paper
have varying column names across versions. We try a list of likely column
candidates per field. Override via --column-map if needed.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

from ..extraction.registry import NA_IDENTIFIER, is_na_identifier
from ..extraction.url_builder import normalize_identifier, normalize_repository

# Per-field, ordered list of column-name candidates we accept.
COLUMN_CANDIDATES = {
    "paper_id": [
        "paper_id", "article_id", "source_article_id",
    ],
    "pmcid": ["pmcid", "PMCID", "pmc_id", "PMC_ID"],
    "pmid": ["pmid", "PMID"],
    "doi": ["doi", "DOI", "paper_doi", "article_doi"],
    "dataset_identifier": [
        "dataset_identifier", "dataset_id", "accession", "accession_number",
        "identifier", "id",
    ],
    "repository": [
        "repository", "repo", "database", "source", "data_repository",
    ],
    "url": ["url", "URL", "link"],
    # Some benchmarks (e.g. DataRef-EXP) only give a citing-publication URL.
    "paper_url": [
        "citing_publication_link", "publication_link", "paper_url",
        "article_url", "source_url",
    ],
}

_PMCID_FROM_URL = re.compile(r"PMC\d{4,}", re.IGNORECASE)
_PMID_FROM_URL = re.compile(r"/pubmed/(\d+)", re.IGNORECASE)
_DOI_FROM_URL = re.compile(r"doi\.org/(10\.\d{4,9}/[\w\.\-/:;()]+)", re.IGNORECASE)


def _pick(row: dict[str, Any], candidates: list[str]) -> str:
    for c in candidates:
        if c in row and row[c] is not None and str(row[c]).strip():
            return str(row[c]).strip()
    return ""


def _read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        delim = "\t" if suffix == ".tsv" else ","
        # utf-8-sig strips a UTF-8 BOM from the header row if present.
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f, delimiter=delim))
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "rows" in data:
            return data["rows"]
        raise ValueError(f"Unrecognized JSON shape in {path}")
    raise ValueError(f"Unsupported groundtruth format: {path}")


def _normalize_doi(doi: str) -> str:
    s = doi.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:", "DOI:"):
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):]
            break
    return s.strip().lower()


def normalize_groundtruth_row(row: dict[str, Any]) -> dict[str, Any]:
    pmcid = _pick(row, COLUMN_CANDIDATES["pmcid"]).upper()
    if pmcid and not pmcid.startswith("PMC"):
        pmcid = "PMC" + pmcid
    pmid = _pick(row, COLUMN_CANDIDATES["pmid"])
    doi = _normalize_doi(_pick(row, COLUMN_CANDIDATES["doi"]))
    paper_url = _pick(row, COLUMN_CANDIDATES["paper_url"])
    if paper_url:
        if not pmcid:
            m = _PMCID_FROM_URL.search(paper_url)
            if m:
                pmcid = m.group(0).upper()
        if not pmid:
            m = _PMID_FROM_URL.search(paper_url)
            if m:
                pmid = m.group(1)
        if not doi:
            m = _DOI_FROM_URL.search(paper_url)
            if m:
                doi = _normalize_doi(m.group(1))
    paper_id = _pick(row, COLUMN_CANDIDATES["paper_id"]) or pmcid or pmid or doi or paper_url

    raw_id = _pick(row, COLUMN_CANDIDATES["dataset_identifier"])
    raw_repo = _pick(row, COLUMN_CANDIDATES["repository"])
    # Collapse "", "N/A", "n/a", "NA", "none" → uniform sentinel "N/A". This
    # preserves rows where the benchmark recorded "this paper has no
    # extractable identifier" so they can match a system that emits N/A
    # instead of being silently dropped.
    if is_na_identifier(raw_id):
        ds_id = NA_IDENTIFIER
    else:
        ds_id = normalize_identifier(raw_id)
    if is_na_identifier(raw_repo):
        repo = NA_IDENTIFIER
    else:
        repo = normalize_repository(raw_repo) or raw_repo.strip()

    return {
        "paper_id": paper_id,
        "pmcid": pmcid,
        "pmid": pmid,
        "paper_doi": doi,
        "dataset_identifier": ds_id,
        "repository": repo,
        "url": _pick(row, COLUMN_CANDIDATES["url"]),
    }


def load_groundtruth(path: Path) -> list[dict[str, Any]]:
    rows = _read_rows(Path(path))
    out: list[dict[str, Any]] = []
    for r in rows:
        norm = normalize_groundtruth_row(r)
        # Keep N/A rows; only skip rows with no usable identifier slot at all
        # (which shouldn't happen post-normalization but guards against
        # entirely empty CSV rows).
        if norm["dataset_identifier"]:
            out.append(norm)
    return out


def group_by_paper(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        key = paper_key(r)
        grouped.setdefault(key, []).append(r)
    return grouped


def paper_key(row: dict[str, Any]) -> str:
    """Canonical paper key used for matching predictions to ground truth."""
    for key in ("pmcid", "pmid", "paper_doi", "paper_id"):
        v = (row.get(key) or "").strip().lower()
        if v:
            return f"{key}:{v}"
    return "paper_id:"
