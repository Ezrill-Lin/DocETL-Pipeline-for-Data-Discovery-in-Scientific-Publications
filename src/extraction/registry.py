"""Registry helpers derived from config/repositories.yaml.

Single source of truth for:
- the regex used to recognize accession-like tokens in candidate passages,
- the regex used to recognize repository names in passages,
- per-repository identifier schema validation,
- synonym groups used by the matcher to treat equivalent repositories
  (e.g., PRIDE / ProteomeXchange) as the same at triple-match time,
- the bullet list of repositories rendered into the LLM prompt.

Adding a repository to repositories.yaml propagates everywhere through
this module — there is no per-repo regex hardcoded elsewhere.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "repositories.yaml"

# Sentinel used by the pipeline when a paper has no extractable identifier
# (either because the paper has no dataset reference, or because it cites a
# repository without giving a specific accession). Compared case-insensitively.
NA_IDENTIFIER = "N/A"
_NA_VARIANTS = {"", "n/a", "na", "none", "null"}


def is_na_identifier(value: str | None) -> bool:
    """True if `value` represents a "no identifier" marker."""
    if value is None:
        return True
    return str(value).strip().lower() in _NA_VARIANTS


@lru_cache(maxsize=1)
def load_repositories() -> list[dict[str, Any]]:
    """Load repository entries from the YAML config."""
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("repositories", []) or []


@lru_cache(maxsize=1)
def all_prefixes() -> tuple[str, ...]:
    """Every accession prefix declared in the config, deduped + sorted longest-first."""
    seen: dict[str, None] = {}
    for repo in load_repositories():
        for prefix in repo.get("prefixes", []) or []:
            if prefix:
                seen[prefix] = None
    # Sort longest-first so multi-char prefixes like 'PRJNA' win over 'P' edges
    return tuple(sorted(seen.keys(), key=lambda p: -len(p)))


@lru_cache(maxsize=1)
def accession_pattern() -> re.Pattern[str]:
    """Compiled regex matching accession-like tokens, derived from prefixes.

    Each prefix is followed by one or more characters from [A-Za-z0-9.] so we
    catch e.g. `phs001249.v1.p1` and `PRJNA306801` without re-tuning the regex
    when a new repo is added.
    """
    prefixes = all_prefixes()
    if not prefixes:
        # Fallback: never match
        return re.compile(r"(?!x)x")
    escaped = "|".join(re.escape(p) for p in prefixes)
    return re.compile(rf"\b(?:{escaped})[A-Za-z0-9.]*\d[A-Za-z0-9.]*\b")


@lru_cache(maxsize=1)
def repository_pattern() -> re.Pattern[str]:
    """Compiled regex matching any known repository name or alias."""
    names: dict[str, None] = {}
    for repo in load_repositories():
        if repo.get("canonical"):
            names[repo["canonical"]] = None
        for alias in repo.get("aliases", []) or []:
            if alias:
                names[alias] = None
    if not names:
        return re.compile(r"(?!x)x")
    # Sort longest-first so 'European Genome Phenome Archive' beats 'EGA' edges
    sorted_names = sorted(names.keys(), key=lambda n: -len(n))
    escaped = "|".join(re.escape(n) for n in sorted_names)
    return re.compile(rf"\b(?:{escaped})\b", re.IGNORECASE)


@lru_cache(maxsize=1)
def alias_to_canonical() -> dict[str, str]:
    """Lowercased alias / canonical → canonical name."""
    out: dict[str, str] = {}
    for repo in load_repositories():
        canonical = repo.get("canonical", "")
        if not canonical:
            continue
        out[canonical.lower()] = canonical
        for alias in repo.get("aliases", []) or []:
            if alias:
                out[alias.lower()] = canonical
    return out


@lru_cache(maxsize=1)
def canonical_to_synonym_group() -> dict[str, str]:
    """Canonical name → synonym-group key. Defaults to lowercased canonical."""
    out: dict[str, str] = {}
    for repo in load_repositories():
        canonical = repo.get("canonical", "")
        if not canonical:
            continue
        group = repo.get("synonym_group") or canonical.lower()
        # Last-write-wins on duplicate canonicals (e.g., GEO has two entries),
        # but groups are defined consistently per canonical so this is safe.
        out[canonical] = group
    return out


def synonym_group(repository: str | None) -> str:
    """Return the synonym-group key for a repository name (any alias).

    Free-form names that don't resolve to a known repository are returned
    lowercased and stripped, so non-registered repositories still compare
    consistently across predictions and groundtruth.
    """
    if not repository:
        return ""
    s = str(repository).strip()
    if not s:
        return ""
    canonical = alias_to_canonical().get(s.lower())
    if canonical:
        return canonical_to_synonym_group().get(canonical, canonical.lower())
    return s.lower()


@lru_cache(maxsize=None)
def _identifier_regex(canonical: str) -> re.Pattern[str] | None:
    """Compile and cache the schema regex for a canonical repository.

    Case-insensitive: real papers and LLMs both write `phs001249` and
    `PHS001249` interchangeably; we want the schema check to accept both
    and let the normalizer settle on a canonical case downstream.
    """
    for repo in load_repositories():
        if repo.get("canonical") == canonical:
            pattern = repo.get("identifier_pattern") or ""
            if pattern:
                return re.compile(pattern, re.IGNORECASE)
    return None


def validate_identifier(identifier: str, canonical_repo: str) -> bool:
    """Check identifier against the schema declared for `canonical_repo`.

    Returns True if no schema is declared (we can't validate, so accept).
    Returns False only when a schema IS declared and the identifier fails.
    """
    if not identifier or not canonical_repo:
        return True
    rx = _identifier_regex(canonical_repo)
    if rx is None:
        return True
    return bool(rx.match(identifier))


def prompt_repository_block(indent: str = "") -> str:
    """Render the repository / prefix bullet list for the LLM prompt.

    Grouped by canonical name so duplicate YAML entries (e.g., GEO with GSE
    and GEO with GDS) collapse into one bullet.

    `indent` is prepended to every line *after the first*. The first line is
    emitted unindented because it inherits the placeholder's indent from the
    surrounding YAML block scalar; subsequent lines need explicit indent so
    the block scalar's column rule is preserved.
    """
    grouped: dict[str, dict[str, list[str]]] = {}
    for repo in load_repositories():
        canonical = repo.get("canonical", "")
        if not canonical:
            continue
        g = grouped.setdefault(canonical, {"prefixes": [], "aliases": []})
        for p in repo.get("prefixes", []) or []:
            if p and p not in g["prefixes"]:
                g["prefixes"].append(p)
        for a in repo.get("aliases", []) or []:
            if a and a not in g["aliases"] and a.lower() != canonical.lower():
                g["aliases"].append(a)

    lines: list[str] = []
    for canonical, info in grouped.items():
        # Suppress DOI-shaped prefixes (e.g. 10.17632) from the prefix list —
        # those are documented under "DOI" form below.
        non_doi_prefixes = [p for p in info["prefixes"] if not p.startswith("10.")]
        if non_doi_prefixes:
            shown = ", ".join(f"{p}####" for p in non_doi_prefixes)
            lines.append(f"  - {canonical}: {shown}")
        else:
            lines.append(f"  - {canonical}: DOI or repository-specific identifier")
    sep = "\n" + indent
    return sep.join(lines)
