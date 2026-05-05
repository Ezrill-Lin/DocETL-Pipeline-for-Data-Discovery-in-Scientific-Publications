"""Fetch papers referenced in REV_sample_groundtruth.csv.

Samples --n unique citing papers from the ground-truth file so that
every fetched paper has at least one evaluable ground-truth record.

Strategy per paper URL:
  1. URL contains a PMC ID  →  fetch JATS XML directly via NCBI efetch.
  2. URL contains a DOI but no PMC ID  →  batch-resolve with NCBI ID Converter;
     if a PMC ID is returned, fetch JATS XML.
  3. No PMC ID after resolution  →  download PDF via Playwright browser automation.

Output:
  XML files go to  data/raw/rev/<PMCID>.xml
  PDF files go to  data/raw/rev/<paper_id>.pdf

The preprocessor (src/preprocess/build_docetl_input.py) handles mixed XML+PDF
directories via rglob, so both formats can coexist in data/raw/rev/ and be
ingested in a single preprocess pass.

Usage:
    uv run python scripts/fetch_rev_papers.py                # all GT papers
    uv run python scripts/fetch_rev_papers.py --no-headless  # show browser
"""
from __future__ import annotations

import argparse
import concurrent.futures
import re
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.preprocess.browser_automation import BrowserAutomationError, download_pdf

GT_PATH = REPO_ROOT / "data" / "benchmark" / "REV_sample_groundtruth.csv"
OUT_DIR = REPO_ROOT / "data" / "raw" / "rev"

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
IDCONV_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
IDCONV_BATCH = 100  # IDs per NCBI ID-converter request

_PMC_RE = re.compile(r"PMC(\d{4,})", re.IGNORECASE)
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[\w.\-/:;()]+)", re.IGNORECASE)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_pmcid(url: str) -> str | None:
    """Return normalised 'PMCxxxxxxx' if the URL encodes a PMC ID."""
    m = _PMC_RE.search(url)
    return f"PMC{m.group(1)}" if m else None


def _extract_doi(url: str) -> str | None:
    """Return bare DOI (without scheme) from a dx.doi.org / doi.org URL."""
    # doi.org redirect: https://dx.doi.org/10.xxxx/yyyy
    for prefix in ("doi.org/", "dx.doi.org/"):
        if prefix in url.lower():
            return url.split(prefix, 1)[1].strip("/")
    m = _DOI_RE.search(url)
    return m.group(1) if m else None


def _paper_slug(url: str) -> str:
    """Fallback readable file stem from a URL."""
    slug = url.rstrip("/").rsplit("/", 1)[-1] or url.rsplit("//", 1)[-1]
    return re.sub(r"[^\w\-.]", "_", slug)[:80]


def _resolve_dois_to_pmc(dois: list[str]) -> dict[str, str]:
    """Batch-resolve DOIs to PMC IDs via NCBI ID Converter.

    Returns {doi: pmcid} for DOIs that have a PMC record.
    Rate: 1 req per batch; polite 0.35 s sleep after each.
    """
    mapping: dict[str, str] = {}
    for i in range(0, len(dois), IDCONV_BATCH):
        batch = dois[i : i + IDCONV_BATCH]
        try:
            r = requests.get(
                IDCONV_URL,
                params={"ids": ",".join(batch), "format": "json", "idtype": "doi"},
                timeout=30,
            )
            r.raise_for_status()
            for rec in r.json().get("records", []):
                doi = rec.get("doi", "")
                pmcid = rec.get("pmcid", "")
                if doi and pmcid:
                    mapping[doi.lower()] = pmcid
        except Exception as e:
            print(f"  [warn] NCBI ID Converter batch failed: {e}", file=sys.stderr)
        time.sleep(0.35)
    return mapping


def _fetch_pmc_xml(pmcid: str, out_dir: Path, label: str) -> str:
    """Download JATS XML for a PMC article. Returns the outcome string."""
    out = out_dir / f"{pmcid}.xml"
    if out.exists() and out.stat().st_size > 1000:
        return f"{label}  (cached xml)"
    numeric_id = pmcid.replace("PMC", "").replace("pmc", "")
    params = {"db": "pmc", "id": numeric_id, "rettype": "xml"}
    delay = 1.0
    for attempt in range(5):
        try:
            r = requests.get(EFETCH_URL, params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(delay)
                delay *= 2
                continue
            r.raise_for_status()
        except requests.HTTPError as e:
            return f"{label}  FAILED (xml): {e}"
        except Exception as e:
            return f"{label}  FAILED (xml): {e}"
        out.write_bytes(r.content)
        time.sleep(0.4)
        return f"{label}  {len(r.content):,} bytes  (xml)"
    return f"{label}  FAILED (xml): too many 429s"


def _fetch_pdf_for(paper_id: str, url: str, out_dir: Path, label: str, headless: bool) -> str:
    """Download PDF via Playwright. Returns the outcome string."""
    out = out_dir / f"{paper_id}.pdf"
    if out.exists() and out.stat().st_size > 1000:
        return f"{label}  (cached pdf)"
    try:
        path = download_pdf(url=url, paper_id=paper_id, raw_dir=out_dir, headless=headless)
        return f"{label}  {path.stat().st_size:,} bytes  (pdf)"
    except BrowserAutomationError as e:
        return f"{label}  FAILED (pdf): {e}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", type=Path, default=GT_PATH,
                   help="Path to REV ground-truth CSV (default: data/benchmark/REV_sample_groundtruth.csv)")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR,
                   help="Output directory for downloaded papers (default: data/raw/rev/)")
    p.add_argument("--workers-xml", type=int, default=3,
                   help="Concurrent workers for XML downloads (default: 3, respects NCBI limit)")
    p.add_argument("--workers-pdf", type=int, default=4,
                   help="Concurrent workers for PDF browser downloads (default: 4)")
    p.add_argument("--no-headless", action="store_true",
                   help="Show browser window during PDF download (debug)")
    args = p.parse_args(argv)

    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas is required. Run: uv add pandas pyarrow", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    headless = not args.no_headless

    # ── 1. Load all unique papers from GT ────────────────────────────────
    print(f"Loading {args.gt.name} …")
    df = pd.read_csv(args.gt)
    rng = (
        df["citing_publication_link"]
        .dropna()
        .unique()
        .tolist()
    )
    print(f"Found {len(rng)} unique papers in ground truth")

    # ── 2. Classify each URL ──────────────────────────────────────────────
    pmc_tasks:  list[tuple[str, str, str]] = []   # (pmcid, url, label)
    doi_tasks:  list[tuple[str, str, str]] = []   # (doi, url, label)
    pdf_tasks:  list[tuple[str, str, str]] = []   # (paper_id, url, label)

    for i, url in enumerate(rng, 1):
        label = f"[{i}/{len(rng)}] {url[:80]}"
        pmcid = _extract_pmcid(url)
        if pmcid:
            pmc_tasks.append((pmcid, url, label))
            continue
        doi = _extract_doi(url)
        if doi:
            doi_tasks.append((doi, url, label))
        else:
            slug = _paper_slug(url)
            pdf_tasks.append((slug, url, label))

    print(f"  Direct PMC: {len(pmc_tasks)}  |  DOI (needs resolution): {len(doi_tasks)}"
          f"  |  Other (PDF): {len(pdf_tasks)}")

    # ── 3. Resolve DOIs → PMC IDs ─────────────────────────────────────────
    if doi_tasks:
        print(f"Resolving {len(doi_tasks)} DOIs via NCBI ID Converter …")
        doi_map = _resolve_dois_to_pmc([doi for doi, _, _ in doi_tasks])
        for doi, url, label in doi_tasks:
            pmcid = doi_map.get(doi.lower())
            if pmcid:
                pmc_tasks.append((pmcid, url, label))
            else:
                slug = _paper_slug(url)
                pdf_tasks.append((slug, url, label))
        resolved = sum(1 for doi, _, _ in doi_tasks if doi_map.get(doi.lower()))
        print(f"  Resolved to PMC: {resolved}  |  Falling back to PDF: {len(doi_tasks) - resolved}")

    print(f"\nFetching {len(pmc_tasks)} XML + {len(pdf_tasks)} PDF papers …")

    # ── 4a. Parallel XML fetch ────────────────────────────────────────────
    if pmc_tasks:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers_xml) as pool:
            futs = {
                pool.submit(_fetch_pmc_xml, pmcid, args.out_dir, label): pmcid
                for pmcid, _, label in pmc_tasks
            }
            for fut in concurrent.futures.as_completed(futs):
                print(fut.result())

    # ── 4b. Parallel PDF fetch ────────────────────────────────────────────
    if pdf_tasks:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers_pdf) as pool:
            futs = {
                pool.submit(_fetch_pdf_for, paper_id, url, args.out_dir, label, headless): paper_id
                for paper_id, url, label in pdf_tasks
            }
            for fut in concurrent.futures.as_completed(futs):
                print(fut.result())

    # ── 5. Summary ────────────────────────────────────────────────────────
    xml_files = list(args.out_dir.glob("*.xml"))
    pdf_files = list(args.out_dir.glob("*.pdf"))
    print(f"\nDone. {len(xml_files)} XML + {len(pdf_files)} PDF files in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
