"""Fetch the PMC articles referenced in EXP_groundtruth.csv.

For each unique PMCID in the ground truth, download the JATS XML from
NCBI's E-utilities (efetch) and save it as data/raw/<PMCID>.xml. JATS XML
parses far better than the HTML reading-room pages.

Run:
    python scripts/fetch_exp_papers.py
    # then preprocess + run the pipeline
    python scripts/prepare_inputs.py --input data/raw --output data/processed/papers.json
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.load_groundtruth import load_groundtruth

GT_PATH = REPO_ROOT / "data" / "benchmark" / "EXP_groundtruth.csv"
OUT_DIR = REPO_ROOT / "data" / "raw"

# NCBI E-utilities efetch returns JATS XML for PMC.
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def main() -> int:
    gt = load_groundtruth(GT_PATH)
    pmcids = sorted({r["pmcid"] for r in gt if r.get("pmcid")})
    print(f"{len(pmcids)} unique PMCIDs to fetch")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, pmcid in enumerate(pmcids, 1):
        out = OUT_DIR / f"{pmcid}.xml"
        if out.exists() and out.stat().st_size > 1000:
            print(f"[{i}/{len(pmcids)}] {pmcid}  (cached)")
            continue
        # PMC efetch wants the integer ID, not the "PMC" prefix.
        numeric_id = pmcid.replace("PMC", "")
        params = {"db": "pmc", "id": numeric_id, "rettype": "xml"}
        try:
            r = requests.get(EFETCH_URL, params=params, timeout=60)
            r.raise_for_status()
        except Exception as e:
            print(f"[{i}/{len(pmcids)}] {pmcid}  FAILED: {e}")
            continue
        out.write_bytes(r.content)
        print(f"[{i}/{len(pmcids)}] {pmcid}  {len(r.content):,} bytes")
        # Be polite to NCBI: ≤ 3 req/s without an API key.
        time.sleep(0.4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
