"""End-to-end orchestrator: prepare → DocETL → evaluate → DataGatherer → report.

Each stage is best-effort: if one stage fails (e.g. no API key, no benchmark
file), later stages are skipped with a clear message and the report still
gets generated from whatever metrics exist.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import yaml

from src.baselines.run_datagatherer import run_datagatherer
from src.evaluation.evaluate import evaluate
from src.extraction.run_docetl import run_pipeline
from src.preprocess.build_docetl_input import build_json_array
from src.reporting.generate_report import generate_report


def _load_settings() -> dict:
    p = REPO_ROOT / "config" / "settings.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> int:
    settings = _load_settings()
    paths = settings.get("paths", {})
    docetl_cfg = settings.get("docetl", {})
    eval_cfg = settings.get("evaluation", {})
    cost_cfg = settings.get("cost", {})

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=REPO_ROOT / paths.get("raw_dir", "data/raw"))
    p.add_argument("--processed", type=Path, default=REPO_ROOT / paths.get("processed_dir", "data/processed") / "papers.json")
    p.add_argument("--predictions", type=Path, default=REPO_ROOT / paths.get("predictions_dir", "data/predictions") / "docetl_predictions.jsonl")
    p.add_argument("--dg-predictions", type=Path, default=REPO_ROOT / paths.get("predictions_dir", "data/predictions") / "datagatherer_predictions.jsonl")
    p.add_argument("--groundtruth", type=Path, default=None, help="Path to a benchmark CSV/JSON. If omitted, evaluation is skipped.")
    p.add_argument("--report", type=Path, default=REPO_ROOT / paths.get("outputs_dir", "outputs") / "report.md")
    p.add_argument("--skip-datagatherer", action="store_true")
    p.add_argument("--model", default=docetl_cfg.get("default_model"))
    args = p.parse_args()

    docetl_summary = None
    dg_summary = None
    docetl_metrics_path = REPO_ROOT / "outputs" / "metrics_docetl.json"
    dg_metrics_path = REPO_ROOT / "outputs" / "metrics_datagatherer.json"

    # 1. Preprocess
    if args.raw_dir.exists() and any(args.raw_dir.iterdir()):
        print(f"[stage 1/5] Preprocessing papers from {args.raw_dir}")
        n = build_json_array(args.raw_dir, args.processed)
        print(f"  → {n} papers in {args.processed}")
    elif args.processed.exists():
        print(f"[stage 1/5] Skipping preprocess; using existing {args.processed}")
    else:
        print(f"[stage 1/5] No raw papers in {args.raw_dir} and no existing processed file. Aborting.")
        return 1

    # 2. DocETL
    print("[stage 2/5] Running DocETL pipeline")
    try:
        docetl_summary = run_pipeline(
            input_path=args.processed,
            output_path=args.predictions,
            model=args.model,
            cost_settings=cost_cfg,
        )
        print(json.dumps(docetl_summary, indent=2))
    except Exception:
        print("[stage 2/5] DocETL run failed:")
        traceback.print_exc()

    # 3. DataGatherer baseline
    if not args.skip_datagatherer:
        print("[stage 3/5] Running DataGatherer baseline")
        try:
            dg_summary = run_datagatherer(args.processed, args.dg_predictions)
            print(json.dumps(dg_summary, indent=2))
        except Exception:
            print("[stage 3/5] DataGatherer run failed:")
            traceback.print_exc()

    # 4. Evaluate
    gt_path = args.groundtruth
    if gt_path is None:
        for candidate in eval_cfg.get("benchmark_files", {}).values():
            cp = REPO_ROOT / candidate
            if cp.exists():
                gt_path = cp
                break
    if gt_path and Path(gt_path).exists():
        print(f"[stage 4/5] Evaluating against {gt_path}")
        if args.predictions.exists():
            try:
                evaluate(args.predictions, gt_path, docetl_metrics_path, label="docetl")
                print(f"  → DocETL metrics → {docetl_metrics_path}")
            except Exception:
                traceback.print_exc()
        if args.dg_predictions.exists():
            try:
                evaluate(args.dg_predictions, gt_path, dg_metrics_path, label="datagatherer")
                print(f"  → DataGatherer metrics → {dg_metrics_path}")
            except Exception:
                traceback.print_exc()
    else:
        print("[stage 4/5] No ground truth available; skipping evaluation.")

    # 5. Report
    print("[stage 5/5] Generating report")
    generate_report(
        docetl_metrics_path=docetl_metrics_path,
        datagatherer_metrics_path=dg_metrics_path if dg_metrics_path.exists() else None,
        docetl_run_summary=docetl_summary,
        datagatherer_run_summary=dg_summary,
        output_path=args.report,
    )
    print(f"  → {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
