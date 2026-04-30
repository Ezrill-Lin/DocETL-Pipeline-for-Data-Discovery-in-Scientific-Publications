"""Deterministic URL construction for extracted dataset identifiers.

Rules:
- The LLM extracts identifiers and (optionally) repository names.
- This module decides the canonical repository, normalizes the identifier,
  and constructs the URL from a hard-coded template.
- If the URL pattern is not known, return an empty string. We never
  fabricate a URL.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "repositories.yaml"


@lru_cache(maxsize=1)
def _load_config() -> list[dict[str, Any]]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("repositories", [])


def normalize_identifier(identifier: str) -> str:
    """Strip whitespace, surrounding quotes/punctuation, and trailing periods.

    Identifiers are normalized to upper case for known accession prefixes
    (GSE, PXD, MSV, ...) and left untouched otherwise (DOIs are kept verbatim
    because they are case-insensitive but often presented in lower case).
    """
    if identifier is None:
        return ""
    s = str(identifier).strip()
    # Iteratively peel off surrounding punctuation/quotes
    junk = "\"'`()[]{}.,;:"
    prev = None
    while prev != s:
        prev = s
        s = s.strip(junk)
    s = re.sub(r"\s+", "", s)
    if s.lower().startswith("doi:"):
        s = s[4:]
    if s.lower().startswith("https://doi.org/"):
        s = s[len("https://doi.org/"):]
    elif s.lower().startswith("http://doi.org/"):
        s = s[len("http://doi.org/"):]
    # Upper-case known accession-style prefixes
    accession_prefixes = (
        "GSE", "GSM", "GPL", "GDS", "PXD", "MSV", "JPST", "IPX", "PASS",
        "E-MTAB", "E-GEOD", "E-PROT", "E-MEXP",
        "SRP", "SRR", "SRX", "SRS", "SRA",
        "PRJNA", "PRJEB", "PRJDB",
        "SAMN", "SAMEA", "SAMD",
        "ERP", "ERR", "ERX", "ERS",
    )
    upper = s.upper()
    for p in accession_prefixes:
        if upper.startswith(p):
            return upper
    return s


def normalize_repository(repository: str | None) -> str:
    """Map any alias of a known repository to its canonical name."""
    if not repository:
        return ""
    s = repository.strip()
    if not s:
        return ""
    target = s.lower()
    for repo in _load_config():
        if repo["canonical"].lower() == target:
            return repo["canonical"]
        for alias in repo.get("aliases", []):
            if alias.lower() == target:
                return repo["canonical"]
    # Loose contains-match as last resort
    for repo in _load_config():
        if repo["canonical"].lower() in target:
            return repo["canonical"]
        for alias in repo.get("aliases", []):
            if alias.lower() in target:
                return repo["canonical"]
    return s  # leave unknown repos untouched (so the user can see them)


_DOI_RE = re.compile(r"^10\.\d{4,9}/", re.IGNORECASE)


def infer_repository(identifier: str, repository: str | None) -> str:
    """Infer the canonical repository from the identifier prefix.

    If a repository is already provided and resolves to a known canonical
    name, prefer that. Otherwise infer from a strongly diagnostic prefix.
    """
    norm_id = normalize_identifier(identifier)
    norm_repo = normalize_repository(repository)
    if norm_repo:
        # Trust an explicit, known repository name
        for repo in _load_config():
            if repo["canonical"] == norm_repo:
                return norm_repo
    # Prefix-driven inference
    for repo in _load_config():
        for prefix in repo.get("prefixes", []):
            if norm_id.upper().startswith(prefix.upper()):
                return repo["canonical"]
    if _DOI_RE.match(norm_id):
        # If repository was something free-form, keep the user-provided string;
        # otherwise label as DOI.
        return norm_repo or "DOI"
    return norm_repo


def build_dataset_url(identifier: str, repository: str | None) -> str:
    """Construct a dataset URL deterministically. Empty string if unknown."""
    norm_id = normalize_identifier(identifier)
    if not norm_id:
        return ""
    canonical = infer_repository(norm_id, repository)
    if not canonical:
        return ""
    is_doi = bool(_DOI_RE.match(norm_id))

    # Try to find a matching repository entry. For prefix-bound entries
    # (e.g. GSE -> GEO accession URL), match the prefix; otherwise match by
    # canonical and prefer entries with empty prefixes.
    candidates = [r for r in _load_config() if r["canonical"] == canonical]
    if not candidates:
        # Unknown repository — but if the identifier itself is a DOI we can
        # still build a doi.org URL deterministically.
        if is_doi:
            return f"https://doi.org/{norm_id}"
        return ""
    # Prefer prefix-matched template
    for repo in candidates:
        for prefix in repo.get("prefixes", []):
            if norm_id.upper().startswith(prefix.upper()):
                tpl = repo.get("url_template") or ""
                if tpl:
                    return tpl.format(id=norm_id)
    # Fallback: first non-prefix template
    for repo in candidates:
        if not repo.get("prefixes"):
            tpl = repo.get("url_template") or ""
            if tpl:
                # DOI template requires a DOI-shaped id
                if "{id}" in tpl and "doi.org" in tpl and not is_doi:
                    return ""
                return tpl.format(id=norm_id)
    return ""
