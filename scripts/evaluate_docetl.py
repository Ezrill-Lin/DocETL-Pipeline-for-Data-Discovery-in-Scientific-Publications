"""Score DocETL predictions against benchmark ground truth."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evaluate import evaluate


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--groundtruth", required=True, type=Path)
    p.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "metrics_docetl.json")
    p.add_argument("--label", default="docetl")
    args = p.parse_args()

    summary = evaluate(args.predictions, args.groundtruth, args.output, label=args.label)
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=2))
    print(f"Full report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
