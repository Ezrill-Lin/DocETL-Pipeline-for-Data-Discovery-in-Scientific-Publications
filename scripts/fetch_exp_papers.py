"""Fetch the papers referenced in EXP_groundtruth.csv.

For PMC articles, the JATS XML is downloaded via NCBI's E-utilities (efetch)
and saved as data/raw/<PMCID>.xml.  JATS XML parses far better than the HTML
reading-room pages.

For non-PMC URLs (publisher pages, DOI redirects, etc.) the PDF is downloaded
via browser automation (Playwright) and saved as data/raw/<paper_id>.pdf.
Playwright + Chromium must be installed:
    uv add playwright
    uv run playwright install chromium

Run:
    python scripts/fetch_exp_papers.py
    # optionally show browser window for debugging:
    python scripts/fetch_exp_papers.py --no-headless
    # then preprocess + run the pipeline
    python scripts/prepare_inputs.py --input data/raw --output data/processed/papers.json
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.load_groundtruth import load_groundtruth
from src.preprocess.browser_automation import download_pdf, BrowserAutomationError

GT_PATH = REPO_ROOT / "data" / "benchmark" / "EXP_groundtruth.csv"
OUT_DIR = REPO_ROOT / "data" / "raw"

# NCBI E-utilities efetch returns JATS XML for PMC.
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

_PMC_RE = re.compile(r"PMC(\d{4,})", re.IGNORECASE)


def _fetch_pmc_xml(pmcid: str, out_dir: Path, index: str) -> None:
    out = out_dir / f"{pmcid}.xml"
    if out.exists() and out.stat().st_size > 1000:
        print(f"[{index}] {pmcid}  (cached xml)")
        return
    numeric_id = pmcid.replace("PMC", "").replace("pmc", "")
    params = {"db": "pmc", "id": numeric_id, "rettype": "xml"}
    try:
        r = requests.get(EFETCH_URL, params=params, timeout=60)
        r.raise_for_status()
    except Exception as e:
        print(f"[{index}] {pmcid}  FAILED (xml): {e}")
        return
    out.write_bytes(r.content)
    print(f"[{index}] {pmcid}  {len(r.content):,} bytes  (xml)")
    # Be polite to NCBI: ≤ 3 req/s without an API key.
    time.sleep(0.4)


def _fetch_pdf(paper_id: str, url: str, out_dir: Path, index: str, headless: bool) -> None:
    out = out_dir / f"{paper_id}.pdf"
    if out.exists() and out.stat().st_size > 1000:
        print(f"[{index}] {paper_id}  (cached pdf)")
        return
    print(f"[{index}] {paper_id}  downloading PDF via browser …")
    try:
        path = download_pdf(
            url=url,
            paper_id=paper_id,
            raw_dir=out_dir,
            headless=headless,
        )
        print(f"[{index}] {paper_id}  {path.stat().st_size:,} bytes  (pdf)")
    except BrowserAutomationError as e:
        print(f"[{index}] {paper_id}  FAILED (pdf): {e}", file=sys.stderr)


def _load_gt_urls(gt_path: Path) -> list[tuple[str, str]]:
    """Return a deduplicated list of (paper_id, url) from the ground truth.

    paper_id is the PMCID when available, otherwise a sanitized slug of the URL.
    """
    seen: dict[str, str] = {}  # paper_id -> url
    with gt_path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            url = (row.get("citing_publication_link") or "").strip()
            if not url:
                continue
            m = _PMC_RE.search(url)
            if m:
                paper_id = f"PMC{m.group(1)}"
            else:
                # Use last path segment of URL as a readable slug
                slug = url.rstrip("/").rsplit("/", 1)[-1] or url.rsplit("//", 1)[-1]
                paper_id = re.sub(r"[^\w\-.]", "_", slug)[:80]
            seen.setdefault(paper_id, url)
    return list(seen.items())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--groundtruth", type=Path, default=GT_PATH,
                   help=f"Path to ground-truth CSV (default: {GT_PATH.relative_to(REPO_ROOT)})")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR,
                   help=f"Output directory (default: {OUT_DIR.relative_to(REPO_ROOT)})")
    p.add_argument("--pdf", action="store_true",
                   help="Download PDFs via browser automation for ALL papers (including PMC). "
                        "Saved as <paper_id>.pdf alongside any .xml files. "
                        "Use this to test the browser automation + PDF parser path.")
    p.add_argument("--no-headless", action="store_true",
                   help="Show the browser window when downloading PDFs (debug mode)")
    args = p.parse_args(argv)

    headless = not args.no_headless
    args.out_dir.mkdir(parents=True, exist_ok=True)

    entries = _load_gt_urls(args.groundtruth)
    pmc_entries = [(pid, url) for pid, url in entries if _PMC_RE.match(pid)]
    pdf_entries = [(pid, url) for pid, url in entries if not _PMC_RE.match(pid)]

    print(f"{len(entries)} unique papers: {len(pmc_entries)} PMC, {len(pdf_entries)} non-PMC")
    if args.pdf:
        print("--pdf: will download PDFs via browser for all papers")

    # Always fetch XML for PMC papers (unless --pdf-only behaviour is wanted)
    if not args.pdf:
        for i, (pmcid, _url) in enumerate(pmc_entries, 1):
            _fetch_pmc_xml(pmcid, args.out_dir, f"{i}/{len(entries)}")

    # PDF downloads: non-PMC papers always, PMC papers only when --pdf is set
    pdf_targets = (entries if args.pdf else pdf_entries)
    offset = 0 if args.pdf else len(pmc_entries)
    for i, (paper_id, url) in enumerate(pdf_targets, 1):
        _fetch_pdf(paper_id, url, args.out_dir, f"{offset+i}/{len(entries)}", headless)

    return 0


if __name__ == "__main__":
    sys.exit(main())
