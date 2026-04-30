"""Fetch a list of paper URLs and emit a DocETL-ready JSON / JSONL file.

Sources of URLs (any combination):
  --url URL [URL ...]            URLs given on the command line
  --urls-file path/to/urls.txt   one URL per line (# comments allowed)
  --from-groundtruth path.csv    pull citing_publication_link column from
                                 a DataRef-style ground-truth CSV

The output file is the same shape produced by scripts/prepare_inputs.py, so
it can be passed directly to scripts/run_pipeline.py.

Examples:
  # Run on a couple of explicit PMC URLs
  python scripts/fetch_urls.py \
      --url https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11252349/ \
      --url https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11425778/ \
      --output data/processed/papers.json

  # Pull every paper referenced by EXP_groundtruth.csv
  python scripts/fetch_urls.py \
      --from-groundtruth data/benchmark/EXP_groundtruth.csv \
      --output data/processed/papers.json \
      --save-raw-to data/raw
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.load_groundtruth import load_groundtruth
from src.preprocess.fetch_url import fetch_papers_from_urls, iter_urls_from_file


def _collect_urls(args: argparse.Namespace) -> list[str]:
    urls: list[str] = []
    if args.url:
        urls.extend(args.url)
    if args.urls_file:
        urls.extend(iter_urls_from_file(args.urls_file))
    if args.from_groundtruth:
        gt_rows = load_groundtruth(args.from_groundtruth)
        # Use `citing_publication_link` if normalize_groundtruth_row preserved
        # it; otherwise reconstruct a PMC URL from the parsed PMCID.
        seen: set[str] = set()
        # We didn't save the raw URL, so synthesize one from the PMCID.
        for r in gt_rows:
            pmcid = r.get("pmcid")
            if pmcid and pmcid not in seen:
                seen.add(pmcid)
                urls.append(f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/")
    # de-dup while preserving order
    seen2: set[str] = set()
    deduped = []
    for u in urls:
        if u not in seen2:
            seen2.add(u)
            deduped.append(u)
    return deduped


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", action="append", help="Article URL (repeatable)")
    p.add_argument("--urls-file", type=Path, help="File with one URL per line")
    p.add_argument("--from-groundtruth", type=Path,
                   help="DataRef ground-truth CSV; PMC URLs are synthesized from each unique PMCID")
    p.add_argument("--output", required=True, type=Path,
                   help="Output JSON or JSONL file (DocETL-ready)")
    p.add_argument("--save-raw-to", type=Path, default=None,
                   help="Optional directory to cache fetched HTML/XML for inspection")
    p.add_argument("--polite-delay", type=float, default=0.4,
                   help="Seconds to sleep between fetches (default 0.4 — NCBI asks for ≤ 3 req/s)")
    args = p.parse_args()

    urls = _collect_urls(args)
    if not urls:
        print("[ERROR] No URLs provided. Use --url, --urls-file, or --from-groundtruth.")
        return 1
    print(f"Fetching {len(urls)} URL(s)…")

    records = fetch_papers_from_urls(
        urls, save_raw_to=args.save_raw_to, polite_delay=args.polite_delay
    )
    if not records:
        print("[ERROR] No papers fetched successfully.")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".jsonl":
        with args.output.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        args.output.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"\nWrote {len(records)} paper records → {args.output}")
    if args.save_raw_to:
        print(f"Cached raw HTML/XML  → {args.save_raw_to}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
