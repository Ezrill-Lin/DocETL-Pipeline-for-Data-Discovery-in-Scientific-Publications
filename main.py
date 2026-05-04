"""End-to-end orchestrator: prepare → RTR + FDR → evaluate both → DataGatherer → report.

Each stage is best-effort: if one stage fails (e.g. no API key, missing
benchmark), the report is still generated from whatever results exist.

Examples
--------
    uv run python main.py                          # EXP benchmark, XML, all stages
    uv run python main.py --benchmark rev          # REV benchmark, all stages
    uv run python main.py --benchmark exp --format pdf   # EXP benchmark, PDF source
    uv run python main.py --benchmark rev --skip-datagatherer
    uv run python main.py --benchmark exp --skip-fetch --skip-preprocess
"""
from __future__ import annotations

import argparse
import concurrent.futures
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

# Lazy imports from scripts/ (added here to avoid circular deps)
def _load_script(name: str) -> Any:
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _download_benchmarks() -> int:
    return _load_script("download_benchmarks").main()


def _fetch_papers(gt_path: Path, out_dir: Path, pdf: bool = False) -> int:
    mod = _load_script("fetch_exp_papers")
    argv = ["--groundtruth", str(gt_path), "--out-dir", str(out_dir)]
    if pdf:
        argv.append("--pdf")
    return mod.main(argv)


def _fetch_rev_papers(gt_path: Path, out_dir: Path) -> int:
    mod = _load_script("fetch_rev_papers")
    argv = ["--gt", str(gt_path), "--out-dir", str(out_dir)]
    return mod.main(argv)


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
    p.add_argument("--benchmark", choices=["exp", "rev"], default="exp",
                   help="Which benchmark to run against: exp (EXP_groundtruth, default) "
                        "or rev (Full_REV_dataset / REV_sample_groundtruth)")
    p.add_argument("--format", choices=["xml", "pdf"], default="xml",
                   help="EXP-only: source paper format to download (default: xml). "
                        "Ignored when --benchmark rev.")
    p.add_argument("--raw-dir", type=Path, default=None,
                   help="Override the raw papers directory. Disables auto-fetch.")
    p.add_argument("--groundtruth", type=Path, default=None,
                   help="Override the benchmark file used for evaluation.")
    p.add_argument("--model", default=None,
                   help="LLM model for DocETL pipelines (overrides settings.yaml)")
    p.add_argument("--skip-preprocess", action="store_true",
                   help="Skip preprocessing; reuse existing papers.json")
    p.add_argument("--skip-fetch", action="store_true",
                   help="Skip auto-fetching benchmark and raw papers")
    p.add_argument("--skip-rtr", action="store_true", help="Skip RTR pipeline")
    p.add_argument("--skip-fdr", action="store_true", help="Skip FDR pipeline")
    p.add_argument("--skip-datagatherer", action="store_true", help="Skip DataGatherer baseline")

    args = p.parse_args(argv)

    # ── Derive paths from --benchmark ─────────────────────────────────────
    raw_base  = REPO_ROOT / paths.get("raw_dir", "data/raw")
    proc_base = REPO_ROOT / paths.get("processed_dir", "data/processed")
    pred_base = REPO_ROOT / paths.get("predictions_dir", "data/predictions")
    bm_dir    = REPO_ROOT / paths.get("benchmark_dir", "data/benchmark")

    # Ground-truth file auto-selected by benchmark
    _gt_defaults = {
        "exp": bm_dir / "EXP_groundtruth.csv",
        "rev": bm_dir / "REV_sample_groundtruth.csv",
    }
    # Parquet for REV fetch
    _rev_parquet = bm_dir / "Full_REV_dataset_citation_records_Table.parquet"

    gt_path = Path(args.groundtruth) if args.groundtruth else _gt_defaults[args.benchmark]

    # raw_dir: --raw-dir > benchmark default
    if args.raw_dir is not None:
        raw_dir = REPO_ROOT / args.raw_dir if not Path(args.raw_dir).is_absolute() else Path(args.raw_dir)
    else:
        raw_dir = raw_base / args.benchmark   # data/raw/exp/  or  data/raw/rev/
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

    # ── Stage 0: Auto-download benchmark + raw papers ─────────────────────
    if args.skip_fetch or args.raw_dir is not None:
        reason = "--raw-dir set" if args.raw_dir is not None else "--skip-fetch set"
        print(f"[stage 0/5] {reason}; skipping auto-download.")
    else:
        # Download benchmark files if missing
        if not gt_path.exists():
            print("[stage 0/5] Downloading benchmark files from Zenodo...")
            try:
                _download_benchmarks()
            except Exception:
                print("  [warn] Benchmark download failed — evaluation will be skipped.")
                traceback.print_exc()
        else:
            print(f"[stage 0/5] Benchmark already present at {gt_path}")

        # Fetch raw papers if raw_dir is empty
        has_papers = (
            raw_dir.exists()
            and any(f for f in raw_dir.iterdir()
                    if f.suffix.lower() in {".xml", ".pdf", ".html", ".htm"}
                    and f.stat().st_size > 100)
        ) if raw_dir.exists() else False

        if not has_papers:
            raw_dir.mkdir(parents=True, exist_ok=True)
            if args.benchmark == "exp":
                fmt = args.format
                print(f"[stage 0/5] Fetching EXP papers ({fmt.upper()}) into {raw_dir} …")
                if fmt == "pdf":
                    print("  (PDF mode: using browser automation via Playwright)")
                try:
                    _fetch_papers(gt_path, raw_dir, pdf=(fmt == "pdf"))
                except Exception:
                    print("  [warn] Paper fetch failed — pipeline may abort at preprocessing.")
                    traceback.print_exc()
            else:  # rev
                if not _rev_parquet.exists():
                    print("  [warn] REV parquet not found — run download_benchmarks first.")
                else:
                    print(f"[stage 0/5] Fetching REV sample papers into {raw_dir} …")
                    try:
                        _fetch_rev_papers(gt_path, raw_dir)
                    except Exception:
                        print("  [warn] REV paper fetch failed — pipeline may abort at preprocessing.")
                        traceback.print_exc()
        else:
            print(f"[stage 0/5] Raw papers already present in {raw_dir}")

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

    # ── Stages 2-3/5: RTR, FDR, DataGatherer (parallel) ──────────────────
    # Each task gets its own DocETL intermediate-cache directory to avoid
    # write conflicts when RTR and FDR run concurrently (both have an
    # operation named 'extract_dataset_references' in step 'extract').
    active: dict[str, tuple[str, Any]] = {}
    if not args.skip_rtr:
        active["rtr"] = (
            "DocETL RTR",
            lambda: run_pipeline(
                input_path=processed,
                output_path=rtr_pred,
                pipeline_yaml=rtr_pipeline,
                model=model,
                intermediate_dir=proc_base / ".docetl_cache_rtr",
                cost_settings=cost_cfg,
            ),
        )
    if not args.skip_fdr:
        active["fdr"] = (
            "DocETL FDR",
            lambda: run_pipeline(
                input_path=processed,
                output_path=fdr_pred,
                pipeline_yaml=fdr_pipeline,
                model=model,
                intermediate_dir=proc_base / ".docetl_cache_fdr",
                cost_settings=cost_cfg,
            ),
        )
    if not args.skip_datagatherer:
        active["dg"] = (
            "DataGatherer",
            lambda: run_datagatherer(processed, dg_pred),
        )

    if args.skip_rtr:
        print("[stage 2a/5] RTR skipped (--skip-rtr)")
    if args.skip_fdr:
        print("[stage 2b/5] FDR skipped (--skip-fdr)")
    if args.skip_datagatherer:
        print("[stage 3/5] DataGatherer skipped (--skip-datagatherer)")

    if active:
        running = ", ".join(label for label, _ in active.values())
        print(f"[stages 2-3/5] Running in parallel: {running}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(active)) as pool:
            futures = {pool.submit(fn): (key, label) for key, (label, fn) in active.items()}
            for future in concurrent.futures.as_completed(futures):
                key, label = futures[future]
                try:
                    summaries[key] = future.result()
                    print(f"\n[{label}] done:")
                    print(json.dumps(summaries[key], indent=2))
                except Exception:
                    print(f"\n[{label}] FAILED:")
                    traceback.print_exc()

    # ── Stage 4: Evaluate ─────────────────────────────────────────────────
    if gt_path.exists():
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
