"""Fetch a scientific paper from a URL and parse it for the pipeline.

Supports three classes of input:

1.  **PMC reading-room URLs** — e.g. ``https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1234567/``
    or ``https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/``.
    We bypass the HTML page and pull JATS XML directly from NCBI's
    E-utilities (``efetch``). JATS gives us a clean section tree, captions,
    and explicit DOI / PMID / PMCID elements — much more reliable than
    scraping the rendered HTML.

2.  **Direct XML / NXML URLs** — e.g. PubMed Central FTP, bioRxiv API.
    We fetch the bytes and feed them to the JATS path of ``parse_html_text``.

3.  **Generic publisher HTML** — anything else with a ``Content-Type`` of
    ``text/html``. We fetch the page and run our HTML section heuristic.

The output is the same record shape produced by ``parse_html_file`` /
``parse_pdf_file``, so it slots into ``build_docetl_input`` unchanged.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import requests

from .build_docetl_input import select_candidate_passages
from .parse_html import parse_html_text

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# PMC reading-room URLs come in a few flavors; this captures all of them.
_PMC_URL_RE = re.compile(
    r"(?:^|/)(?:pmc/articles/|articles/|pmc/)PMC(\d{4,})/?",
    re.IGNORECASE,
)

DEFAULT_HEADERS = {
    # A descriptive UA plays nicer with publisher rate limiters than the
    # default ``python-requests/x.y.z``.
    "User-Agent": (
        "docetl-dataref-extractor/0.1 "
        "(+research; contact: see config/settings.yaml)"
    ),
    "Accept": (
        "application/xml, text/xml, application/jats+xml, "
        "text/html;q=0.9, */*;q=0.5"
    ),
}


class FetchError(RuntimeError):
    pass


def _extract_pmcid_from_url(url: str) -> str | None:
    m = _PMC_URL_RE.search(url)
    if not m:
        return None
    return f"PMC{m.group(1)}"


def fetch_jats_for_pmcid(pmcid: str, timeout: int = 60) -> str:
    """Fetch the JATS XML for a PMCID via NCBI E-utilities."""
    numeric = pmcid.upper().replace("PMC", "")
    if not numeric.isdigit():
        raise FetchError(f"Invalid PMCID: {pmcid!r}")
    params = {"db": "pmc", "id": numeric, "rettype": "xml"}
    r = requests.get(EFETCH_URL, params=params, headers=DEFAULT_HEADERS, timeout=timeout)
    r.raise_for_status()
    text = r.text
    # efetch occasionally returns a tiny "no result" envelope; sanity-check.
    if "<article" not in text:
        raise FetchError(f"PMC efetch returned no article body for {pmcid}")
    return text


def _looks_like_xml(content_type: str, body: str) -> bool:
    if "xml" in content_type.lower():
        return True
    head = body.lstrip()[:500].lower()
    return head.startswith("<?xml") or "<article" in head[:2000]


def fetch_paper_from_url(
    url: str,
    save_raw_to: Path | None = None,
    timeout: int = 60,
    polite_delay: float = 0.0,
) -> dict[str, Any]:
    """Fetch and parse one paper from a URL.

    Args:
        url: Article URL (PMC reading room, JATS XML, or generic HTML).
        save_raw_to: Optional directory; when set, the fetched bytes are
            cached to ``<dir>/<paper_id>.<ext>`` for later inspection.
        timeout: Per-request timeout in seconds.
        polite_delay: Sleep this many seconds before fetching. Useful when
            iterating over many PMC URLs (NCBI asks for ≤ 3 req/s).

    Returns:
        A paper record with the same shape as ``parse_html_file`` plus
        ``candidate_passages`` and ``candidate_char_count``.
    """
    if polite_delay:
        time.sleep(polite_delay)

    pmcid = _extract_pmcid_from_url(url)
    raw: str
    fetch_url = url

    try:
        if pmcid:
            # Always prefer JATS XML for PMC articles — far more structured.
            raw = fetch_jats_for_pmcid(pmcid, timeout=timeout)
            ext = "xml"
            paper_id_hint = pmcid
        else:
            r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
            r.raise_for_status()
            raw = r.text
            ext = "xml" if _looks_like_xml(r.headers.get("Content-Type", ""), raw) else "html"
            host = urlparse(url).hostname or "remote"
            paper_id_hint = f"{host}_{abs(hash(url)) % 10**8}"
    except requests.RequestException as e:
        raise FetchError(f"Failed to fetch {url}: {e}") from e

    record = parse_html_text(raw, source=url, default_paper_id=paper_id_hint)
    if pmcid and not record.get("pmcid"):
        record["pmcid"] = pmcid
        if not record.get("paper_id"):
            record["paper_id"] = pmcid

    if save_raw_to:
        save_dir = Path(save_raw_to)
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"{record.get('paper_id') or paper_id_hint}.{ext}"
        out.write_text(raw, encoding="utf-8")
        record["source_path"] = str(out)

    record["source_url"] = url
    record["candidate_passages"] = select_candidate_passages(record)
    record["candidate_char_count"] = len(record["candidate_passages"])
    return record


def iter_urls_from_file(path: Path) -> Iterable[str]:
    """Yield URLs from a plain text file (one per line, # for comments)."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            yield line


def fetch_papers_from_urls(
    urls: Iterable[str],
    save_raw_to: Path | None = None,
    polite_delay: float = 0.4,
) -> list[dict[str, Any]]:
    """Fetch and parse a list of URLs, skipping (with a warning) any that fail."""
    out: list[dict[str, Any]] = []
    for url in urls:
        try:
            rec = fetch_paper_from_url(
                url, save_raw_to=save_raw_to, polite_delay=polite_delay
            )
            out.append(rec)
            print(f"  ✓ {url}  →  {rec.get('paper_id', '?')}  "
                  f"({rec.get('candidate_char_count', 0):,} candidate chars)")
        except FetchError as e:
            print(f"  ✗ {url}  →  {e}")
    return out
