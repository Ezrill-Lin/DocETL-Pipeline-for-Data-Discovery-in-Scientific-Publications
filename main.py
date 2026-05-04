"""End-to-end orchestrator: prepare → RTR + FDR + DataGatherer → evaluate → report.

Supports multi-benchmark and multi-model runs in one invocation. Each
(benchmark, model) combination is fully evaluated and all results are merged
into a single report with a unified Dataset × Model × Method table.

Each stage is best-effort: if one stage fails (e.g. no API key, missing
benchmark), the report is still generated from whatever results exist.

Examples
--------
    uv run python main.py                                          # EXP, xml, default model
    uv run python main.py --benchmark rev                         # REV benchmark
    uv run python main.py --format pdf                            # EXP, PDF source
    uv run python main.py --models gpt-4o-mini,gemini/gemini-2.0-flash
    uv run python main.py --benchmarks exp,rev --models gpt-4o-mini,gemini/gemini-2.0-flash
    uv run python main.py --benchmark exp --skip-datagatherer
    uv run python main.py --benchmark exp --skip-fetch --skip-preprocess
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.baselines.run_datagatherer import run_datagatherer
from src.evaluation.evaluate import evaluate
from src.extraction.run_docetl import run_pipeline
from src.preprocess.build_docetl_input import build_json_array
from src.reporting.generate_report import generate_report


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


def _safe_model(model: str) -> str:
    """Convert a model string to a filesystem-safe name."""
    return model.replace("/", "-").replace(":", "-")


def _run_evaluate(
    predictions: Path,
    gt_path: Path,
    metrics_out: Path,
    label: str,
    *,
    model: str = "",
    benchmark: str = "",
) -> None:
    if predictions.exists():
        try:
            evaluate(predictions, gt_path, metrics_out, label=label, model=model, benchmark=benchmark)
            print(f"    → {label} metrics → {metrics_out.name}")
        except Exception:
            traceback.print_exc()
    else:
        print(f"    [skip] {label}: no predictions file at {predictions}")


def _stage0_fetch(
    benchmark: str,
    gt_path: Path,
    raw_dir: Path,
    rev_parquet: Path,
    fmt: str,
    prefix: str,
) -> None:
    """Download benchmark + raw papers if not already present."""
    if not gt_path.exists():
        print(f"{prefix} Downloading benchmark files from Zenodo...")
        try:
            _download_benchmarks()
        except Exception:
            print(f"  [warn] Benchmark download failed — evaluation will be skipped.")
            traceback.print_exc()
    else:
        print(f"{prefix} Benchmark already present at {gt_path.name}")

    has_papers = (
        raw_dir.exists()
        and any(
            f for f in raw_dir.iterdir()
            if f.suffix.lower() in {".xml", ".pdf", ".html", ".htm"}
            and f.stat().st_size > 100
        )
    ) if raw_dir.exists() else False

    if not has_papers:
        raw_dir.mkdir(parents=True, exist_ok=True)
        if benchmark == "exp":
            print(f"{prefix} Fetching EXP papers ({fmt.upper()}) into {raw_dir} …")
            try:
                _fetch_papers(gt_path, raw_dir, pdf=(fmt == "pdf"))
            except Exception:
                print("  [warn] Paper fetch failed — pipeline may abort at preprocessing.")
                traceback.print_exc()
        else:
            if not rev_parquet.exists():
                print(f"  [warn] REV parquet not found — run download_benchmarks first.")
            else:
                print(f"{prefix} Fetching REV sample papers into {raw_dir} …")
                try:
                    _fetch_rev_papers(gt_path, raw_dir)
                except Exception:
                    print("  [warn] REV paper fetch failed — pipeline may abort at preprocessing.")
                    traceback.print_exc()
    else:
        print(f"{prefix} Raw papers already present in {raw_dir}")


def main(argv: list[str] | None = None) -> int:
    settings = _load_settings()
    paths    = settings.get("paths", {})
    docetl_cfg = settings.get("docetl", {})
    cost_cfg   = settings.get("cost", {})

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── Single-value flags (original interface) ────────────────────────────
    p.add_argument("--benchmark", choices=["exp", "rev"], default="exp",
                   help="Benchmark to run: exp (default) or rev. Use --benchmarks for multiple.")
    p.add_argument("--format", choices=["xml", "pdf"], default="xml",
                   help="EXP-only: source paper format (default: xml).")
    p.add_argument("--model", default=None,
                   help="LLM model for DocETL + DataGatherer (overrides settings.yaml). "
                        "Use --models for multiple.")
    # ── Multi-value flags (new) ────────────────────────────────────────────
    p.add_argument("--benchmarks", default=None,
                   help="Comma-separated benchmarks to run, e.g. 'exp,rev'. "
                        "Overrides --benchmark.")
    p.add_argument("--models", default=None,
                   help="Comma-separated LLM models, e.g. 'gpt-4o-mini,gemini/gemini-2.0-flash'. "
                        "Overrides --model. Each model is run against every benchmark.")
    # ── Path overrides ─────────────────────────────────────────────────────
    p.add_argument("--raw-dir", type=Path, default=None,
                   help="Override the raw papers directory. Disables auto-fetch.")
    p.add_argument("--groundtruth", type=Path, default=None,
                   help="Override the benchmark file used for evaluation.")
    # ── Skip flags ─────────────────────────────────────────────────────────
    p.add_argument("--skip-preprocess", action="store_true",
                   help="Skip preprocessing; reuse existing papers.json.")
    p.add_argument("--skip-fetch", action="store_true",
                   help="Skip auto-fetching benchmark and raw papers.")
    p.add_argument("--skip-rtr", action="store_true", help="Skip RTR pipeline.")
    p.add_argument("--skip-fdr", action="store_true", help="Skip FDR pipeline.")
    p.add_argument("--skip-datagatherer", action="store_true", help="Skip DataGatherer baseline.")

    args = p.parse_args(argv)

    # ── Resolve benchmark list and model list ──────────────────────────────
    benchmark_list: list[str] = (
        [b.strip() for b in args.benchmarks.split(",")]
        if args.benchmarks
        else [args.benchmark]
    )
    _default_model = args.model or docetl_cfg.get("default_model", "gemini/gemini-2.5-flash")
    model_list: list[str] = (
        [m.strip() for m in args.models.split(",")]
        if args.models
        else [_default_model]
    )

    # ── Base paths ─────────────────────────────────────────────────────────
    raw_base    = REPO_ROOT / paths.get("raw_dir", "data/raw")
    proc_base   = REPO_ROOT / paths.get("processed_dir", "data/processed")
    pred_base   = REPO_ROOT / paths.get("predictions_dir", "data/predictions")
    bm_dir      = REPO_ROOT / paths.get("benchmark_dir", "data/benchmark")
    outputs_dir = REPO_ROOT / paths.get("outputs_dir", "outputs")

    _gt_defaults = {
        "exp": bm_dir / "EXP_groundtruth.csv",
        "rev": bm_dir / "REV_sample_groundtruth.csv",
    }
    _rev_parquet = bm_dir / "Full_REV_dataset_citation_records_Table.parquet"

    rtr_pipeline = REPO_ROOT / docetl_cfg.get("pipeline_yaml", "pipelines/pipeline_rtr.yaml")
    fdr_pipeline = REPO_ROOT / "pipelines" / "pipeline_fdr.yaml"

    outputs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = outputs_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_path = outputs_dir / "report.md"

    n_total = len(benchmark_list) * len(model_list)
    print(
        f"Running {len(benchmark_list)} benchmark(s) × {len(model_list)} model(s) "
        f"= {n_total} evaluation run(s)."
    )
    if len(benchmark_list) > 1 or len(model_list) > 1:
        print(f"  Benchmarks: {benchmark_list}")
        print(f"  Models:     {model_list}")

    all_run_summaries: list[dict[str, Any]] = []

    # ══════════════════════════════════════════════════════════════════════
    # Outer loop: benchmarks  (fetch + preprocess once per benchmark)
    # ══════════════════════════════════════════════════════════════════════
    for benchmark in benchmark_list:
        gt_path = Path(args.groundtruth) if args.groundtruth else _gt_defaults[benchmark]

        if args.raw_dir is not None:
            raw_dir = (
                REPO_ROOT / args.raw_dir
                if not Path(args.raw_dir).is_absolute()
                else Path(args.raw_dir)
            )
        else:
            raw_dir = raw_base / benchmark

        # papers.json is per-benchmark (different papers for EXP vs REV)
        processed = proc_base / benchmark / "papers.json"

        bm_prefix = f"[{benchmark.upper()}]"

        # ── Stage 0: Auto-download ─────────────────────────────────────────
        if args.skip_fetch or args.raw_dir is not None:
            reason = "--raw-dir set" if args.raw_dir is not None else "--skip-fetch set"
            print(f"{bm_prefix} [stage 0] {reason}; skipping auto-download.")
        else:
            print(f"{bm_prefix} [stage 0] Checking benchmark + raw papers …")
            _stage0_fetch(benchmark, gt_path, raw_dir, _rev_parquet, args.format, f"  {bm_prefix}")

        # ── Stage 1: Preprocess ────────────────────────────────────────────
        print(f"{bm_prefix} [stage 1] Preprocessing papers")
        if args.skip_preprocess:
            if not processed.exists():
                print(
                    f"  ERROR: --skip-preprocess set but {processed} does not exist. "
                    "Aborting."
                )
                return 1
            print(f"  skipping; using existing {processed}")
        elif raw_dir.exists() and any(raw_dir.rglob("*.*")):
            processed.parent.mkdir(parents=True, exist_ok=True)
            n = build_json_array(raw_dir, processed)
            print(f"  → {n} papers in {processed}")
        elif processed.exists():
            print(f"  skipping; using existing {processed}")
        else:
            print(f"  ERROR: no raw papers in {raw_dir} and no processed file. Aborting.")
            return 1

        # ══════════════════════════════════════════════════════════════════
        # Inner loop: models  (pipelines + evaluate per model)
        # ══════════════════════════════════════════════════════════════════
        for model in model_list:
            sm = _safe_model(model)
            run_prefix = f"[{benchmark.upper()}  {model}]"

            pred_dir = pred_base / benchmark / sm
            pred_dir.mkdir(parents=True, exist_ok=True)

            rtr_pred    = pred_dir / "rtr_predictions.jsonl"
            fdr_pred    = pred_dir / "fdr_predictions.jsonl"
            dg_rtr_pred = pred_dir / "datagatherer_rtr_predictions.jsonl"
            dg_fdr_pred = pred_dir / "datagatherer_fdr_predictions.jsonl"

            m_rtr    = metrics_dir / f"{benchmark}_{sm}_rtr.json"
            m_fdr    = metrics_dir / f"{benchmark}_{sm}_fdr.json"
            m_dg_rtr = metrics_dir / f"{benchmark}_{sm}_datagatherer_rtr.json"
            m_dg_fdr = metrics_dir / f"{benchmark}_{sm}_datagatherer_fdr.json"

            summaries: dict[str, Any] = {"rtr": None, "fdr": None, "dg_rtr": None, "dg_fdr": None}

            # ── Stages 2-3: RTR + FDR + DataGatherer (parallel) ───────────
            active: dict[str, tuple[str, Any]] = {}
            if not args.skip_rtr:
                # Capture loop variables explicitly to avoid closure issues
                _proc, _rtr_pred, _rtr_pipe, _model = processed, rtr_pred, rtr_pipeline, model
                _cache = proc_base / f".docetl_cache_{benchmark}_{sm}_rtr"
                active["rtr"] = (
                    "DocETL RTR",
                    lambda p=_proc, o=_rtr_pred, y=_rtr_pipe, m=_model, c=_cache: run_pipeline(
                        input_path=p, output_path=o, pipeline_yaml=y,
                        model=m, intermediate_dir=c, cost_settings=cost_cfg,
                    ),
                )
            if not args.skip_fdr:
                _proc, _fdr_pred, _fdr_pipe, _model = processed, fdr_pred, fdr_pipeline, model
                _cache = proc_base / f".docetl_cache_{benchmark}_{sm}_fdr"
                active["fdr"] = (
                    "DocETL FDR",
                    lambda p=_proc, o=_fdr_pred, y=_fdr_pipe, m=_model, c=_cache: run_pipeline(
                        input_path=p, output_path=o, pipeline_yaml=y,
                        model=m, intermediate_dir=c, cost_settings=cost_cfg,
                    ),
                )
            if not args.skip_datagatherer:
                _proc, _dg_rtr_pred, _model = processed, dg_rtr_pred, model
                active["dg_rtr"] = (
                    "DataGatherer RTR",
                    lambda p=_proc, o=_dg_rtr_pred, m=_model: run_datagatherer(
                        p, o, strategy="retrieve", llm_name=m,
                    ),
                )
                _proc, _dg_fdr_pred, _model = processed, dg_fdr_pred, model
                active["dg_fdr"] = (
                    "DataGatherer FDR",
                    lambda p=_proc, o=_dg_fdr_pred, m=_model: run_datagatherer(
                        p, o, strategy="full", llm_name=m,
                    ),
                )

            if args.skip_rtr:
                print(f"{run_prefix} [stage 2a] RTR skipped (--skip-rtr)")
            if args.skip_fdr:
                print(f"{run_prefix} [stage 2b] FDR skipped (--skip-fdr)")
            if args.skip_datagatherer:
                print(f"{run_prefix} [stage 3]  DataGatherer (RTR+FDR) skipped (--skip-datagatherer)")

            if active:
                running = ", ".join(label for label, _ in active.values())
                print(f"{run_prefix} [stages 2-3] Running in parallel: {running}")
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(active)) as pool:
                    futures = {pool.submit(fn): (key, lbl) for key, (lbl, fn) in active.items()}
                    for future in concurrent.futures.as_completed(futures):
                        key, lbl = futures[future]
                        try:
                            summaries[key] = future.result()
                            print(f"\n  [{lbl}] done:")
                            print(json.dumps(summaries[key], indent=4))
                        except Exception:
                            print(f"\n  [{lbl}] FAILED:")
                            traceback.print_exc()

            # ── Stage 4: Evaluate ──────────────────────────────────────────
            if gt_path.exists():
                print(f"{run_prefix} [stage 4] Evaluating against {gt_path.name}")
                _run_evaluate(rtr_pred,    gt_path, m_rtr,    "rtr",              model=model, benchmark=benchmark)
                _run_evaluate(fdr_pred,    gt_path, m_fdr,    "fdr",              model=model, benchmark=benchmark)
                _run_evaluate(dg_rtr_pred, gt_path, m_dg_rtr, "datagatherer_rtr", model=model, benchmark=benchmark)
                _run_evaluate(dg_fdr_pred, gt_path, m_dg_fdr, "datagatherer_fdr", model=model, benchmark=benchmark)
            else:
                print(f"{run_prefix} [stage 4] No ground truth at {gt_path}; skipping evaluation.")

            all_run_summaries.append({
                "benchmark": benchmark,
                "model": model,
                "rtr":    summaries["rtr"],
                "fdr":    summaries["fdr"],
                "dg_rtr": summaries["dg_rtr"],
                "dg_fdr": summaries["dg_fdr"],
            })

    # ── Stage 5: Report ────────────────────────────────────────────────────
    print("[stage 5] Generating report")
    generate_report(
        metrics_dir=metrics_dir,
        run_summaries=all_run_summaries,
        output_path=report_path,
    )
    print(f"  → {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
