"""End-to-end orchestrator: prepare → RTR + FDR → evaluate both → DataGatherer → report.

Each stage is best-effort: if one stage fails (e.g. no API key, missing
benchmark), the report is still generated from whatever results exist.

Examples
--------
    uv run python main.py                     # XML papers, all stages
    uv run python main.py --format pdf        # PDF papers, all stages
    uv run python main.py --skip-datagatherer # skip DataGatherer baseline
    uv run python main.py --skip-rtr          # FDR only
    uv run python main.py --skip-fdr          # RTR only
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.baselines.run_datagatherer import run_datagatherer
from src.evaluation.evaluate import evaluate
from src.extraction.run_docetl import run_pipeline
from src.preprocess.build_docetl_input import build_json_array
from src.reporting.generate_report import generate_report


def _load_settings() -> dict:
    p = REPO_ROOT / "config" / "settings.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}


def _run_evaluate(predictions: Path, gt_path: Path, metrics_out: Path, label: str) -> None:
    if predictions.exists():
        try:
            evaluate(predictions, gt_path, metrics_out, label=label)
            print(f"  → {label} metrics → {metrics_out}")
        except Exception:
            traceback.print_exc()
    else:
        print(f"  [skip] {label}: no predictions file at {predictions}")


def main(argv: list[str] | None = None) -> int:
    settings = _load_settings()
    paths = settings.get("paths", {})
    docetl_cfg = settings.get("docetl", {})
    eval_cfg = settings.get("evaluation", {})
    cost_cfg = settings.get("cost", {})

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--format", choices=["xml", "pdf"], default="xml",
                   help="Source paper format: xml (default) or pdf")
    p.add_argument("--groundtruth", type=Path, default=None,
                   help="Benchmark CSV for evaluation (auto-detected from settings if omitted)")
    p.add_argument("--model", default=None,
                   help="LLM model for DocETL pipelines (overrides settings.yaml, e.g. gpt-4o-mini)")
    p.add_argument("--skip-preprocess", action="store_true",
                   help="Skip preprocessing and reuse existing papers.json")
    p.add_argument("--skip-rtr", action="store_true", help="Skip RTR pipeline")
    p.add_argument("--skip-fdr", action="store_true", help="Skip FDR pipeline")
    p.add_argument("--skip-datagatherer", action="store_true", help="Skip DataGatherer baseline")

    args = p.parse_args(argv)

    # Derive all paths from settings + format; nothing is hardcoded in the CLI.
    raw_base  = REPO_ROOT / paths.get("raw_dir", "data/raw")
    proc_base = REPO_ROOT / paths.get("processed_dir", "data/processed")
    pred_base = REPO_ROOT / paths.get("predictions_dir", "data/predictions")
    fmt = args.format

    raw_dir     = raw_base / fmt
    processed   = proc_base / "papers.json"
    rtr_pred    = pred_base / "rtr_predictions.jsonl"
    fdr_pred    = pred_base / "fdr_predictions.jsonl"
    dg_pred     = pred_base / "datagatherer_predictions.jsonl"
    report_path = REPO_ROOT / paths.get("outputs_dir", "outputs") / "report.md"

    rtr_pipeline = REPO_ROOT / docetl_cfg.get("pipeline_yaml",
                       "pipelines/pipeline_rtr.yaml")
    fdr_pipeline = REPO_ROOT / "pipelines" / "pipeline_fdr.yaml"
    model        = args.model or docetl_cfg.get("default_model", "gemini/gemini-2.5-flash")

    outputs_dir = REPO_ROOT / paths.get("outputs_dir", "outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "rtr":          outputs_dir / "metrics_rtr.json",
        "fdr":          outputs_dir / "metrics_fdr.json",
        "datagatherer": outputs_dir / "metrics_datagatherer.json",
    }
    summaries: dict[str, dict | None] = {"rtr": None, "fdr": None, "dg": None}

    # ── Stage 1: Preprocess ────────────────────────────────────────────────
    print("[stage 1/5] Preprocessing papers")
    if args.skip_preprocess:
        if not processed.exists():
            print(f"  ERROR: --skip-preprocess set but {processed} does not exist. Aborting.")
            return 1
        print(f"  skipping; using existing {processed}")
    elif raw_dir.exists() and any(raw_dir.rglob("*.*")):
        n = build_json_array(raw_dir, processed)
        print(f"  → {n} papers in {processed}")
    elif processed.exists():
        print(f"  skipping; using existing {processed}")
    else:
        print(f"  ERROR: no raw papers in {raw_dir} and no processed file. Aborting.")
        return 1

    # ── Stage 2a: RTR pipeline ─────────────────────────────────────────────
    if not args.skip_rtr:
        print("[stage 2a/5] Running DocETL RTR pipeline")
        try:
            summaries["rtr"] = run_pipeline(
                input_path=processed,
                output_path=rtr_pred,
                pipeline_yaml=rtr_pipeline,
                model=model,
                cost_settings=cost_cfg,
            )
            print(json.dumps(summaries["rtr"], indent=2))
        except Exception:
            print("  RTR run failed:")
            traceback.print_exc()
    else:
        print("[stage 2a/5] RTR skipped (--skip-rtr)")

    # ── Stage 2b: FDR pipeline ─────────────────────────────────────────────
    if not args.skip_fdr:
        print("[stage 2b/5] Running DocETL FDR pipeline")
        try:
            summaries["fdr"] = run_pipeline(
                input_path=processed,
                output_path=fdr_pred,
                pipeline_yaml=fdr_pipeline,
                model=model,
                cost_settings=cost_cfg,
            )
            print(json.dumps(summaries["fdr"], indent=2))
        except Exception:
            print("  FDR run failed:")
            traceback.print_exc()
    else:
        print("[stage 2b/5] FDR skipped (--skip-fdr)")

    # ── Stage 3: DataGatherer baseline ────────────────────────────────────
    if not args.skip_datagatherer:
        print("[stage 3/5] Running DataGatherer baseline")
        try:
            summaries["dg"] = run_datagatherer(processed, dg_pred)
            print(json.dumps(summaries["dg"], indent=2))
        except Exception:
            print("  DataGatherer run failed:")
            traceback.print_exc()
    else:
        print("[stage 3/5] DataGatherer skipped (--skip-datagatherer)")

    # ── Stage 4: Evaluate ─────────────────────────────────────────────────
    gt_path = args.groundtruth
    if gt_path is None:
        for candidate in eval_cfg.get("benchmark_files", {}).values():
            cp = REPO_ROOT / candidate
            if cp.exists():
                gt_path = cp
                break
    if gt_path and Path(gt_path).exists():
        print(f"[stage 4/5] Evaluating against {gt_path}")
        _run_evaluate(rtr_pred, gt_path, metrics["rtr"], "rtr")
        _run_evaluate(fdr_pred, gt_path, metrics["fdr"], "fdr")
        _run_evaluate(dg_pred, gt_path, metrics["datagatherer"], "datagatherer")
    else:
        print("[stage 4/5] No ground truth available; skipping evaluation.")

    # ── Stage 5: Report ───────────────────────────────────────────────────
    print("[stage 5/5] Generating report")
    generate_report(
        rtr_metrics_path=metrics["rtr"],
        fdr_metrics_path=metrics["fdr"],
        datagatherer_metrics_path=metrics["datagatherer"] if metrics["datagatherer"].exists() else None,
        rtr_run_summary=summaries["rtr"],
        fdr_run_summary=summaries["fdr"],
        datagatherer_run_summary=summaries["dg"],
        output_path=report_path,
    )
    print(f"  → {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
