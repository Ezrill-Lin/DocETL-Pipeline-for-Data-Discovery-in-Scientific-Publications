"""Convert a directory of HTML/PDF papers into a DocETL-ready JSON file."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.preprocess.build_docetl_input import build_json_array, build_jsonl


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path, help="Directory of HTML/PDF papers")
    p.add_argument("--output", required=True, type=Path, help="Output JSON or JSONL path")
    args = p.parse_args()

    if args.output.suffix.lower() == ".jsonl":
        n = build_jsonl(args.input, args.output)
    else:
        n = build_json_array(args.input, args.output)
    print(f"Wrote {n} paper records to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
