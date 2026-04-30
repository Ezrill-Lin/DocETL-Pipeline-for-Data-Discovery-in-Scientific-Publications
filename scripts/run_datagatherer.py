"""Run DataGatherer as a baseline and write predictions in the DocETL schema."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.baselines.run_datagatherer import run_datagatherer


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path,
                   help="papers.json (output of scripts/prepare_inputs.py or fetch_urls.py)")
    p.add_argument("--output", required=True, type=Path,
                   help="Predictions JSONL path")
    p.add_argument("--strategy", choices=("retrieve", "full"), default="retrieve",
                   help="retrieve = Retrieve-Then-Read (default); full = Full-Document Read")
    p.add_argument("--model", default=None,
                   help="LLM name (defaults to DOCETL_MODEL or gpt-4o-mini)")
    p.add_argument("--summary-out", type=Path, default=REPO_ROOT / "outputs" / "run_datagatherer.json")
    args = p.parse_args()

    status = run_datagatherer(
        input_path=args.input,
        output_path=args.output,
        strategy=args.strategy,
        llm_name=args.model,
    )
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0 if status.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
