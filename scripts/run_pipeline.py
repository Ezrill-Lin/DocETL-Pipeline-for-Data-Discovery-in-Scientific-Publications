"""Run the DocETL extraction pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.extraction.run_docetl import run_pipeline


def _load_settings() -> dict:
    p = REPO_ROOT / "config" / "settings.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> int:
    settings = _load_settings()
    docetl_cfg = settings.get("docetl", {})
    cost_cfg = settings.get("cost", {})

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path, help="papers.json from prepare_inputs.py")
    p.add_argument("--output", required=True, type=Path, help="Final predictions JSONL path")
    p.add_argument("--pipeline", type=Path, default=REPO_ROOT / docetl_cfg.get("pipeline_yaml", "pipelines/dataset_reference_extraction.yaml"))
    p.add_argument("--model", default=docetl_cfg.get("default_model"))
    p.add_argument("--intermediate-dir", type=Path, default=REPO_ROOT / docetl_cfg.get("intermediate_dir", "data/processed/.docetl_cache"))
    p.add_argument("--cost-summary", type=Path, default=REPO_ROOT / "outputs" / "cost_docetl.json")
    args = p.parse_args()

    summary = run_pipeline(
        input_path=args.input,
        output_path=args.output,
        pipeline_yaml=args.pipeline,
        model=args.model,
        intermediate_dir=args.intermediate_dir,
        cost_settings=cost_cfg,
    )
    args.cost_summary.parent.mkdir(parents=True, exist_ok=True)
    args.cost_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
