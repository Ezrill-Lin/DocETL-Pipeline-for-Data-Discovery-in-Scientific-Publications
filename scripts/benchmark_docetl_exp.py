"""Run DocETL RTR and FDR on the EXP benchmark N times and report averaged metrics.

Each run uses a fresh LLM cache (both the DocETL general cache and the
SQLite-backed LLM response cache are wiped before every run) so that we get
genuinely independent samples from the model rather than replaying the same
cached responses 20 times.

Intermediate DocETL checkpoint directories are also removed between runs so
that each run re-processes all papers from scratch.

Usage
-----
    uv run python scripts/benchmark_docetl_exp.py
    uv run python scripts/benchmark_docetl_exp.py --runs 10 --max-threads 10
    uv run python scripts/benchmark_docetl_exp.py --skip-rtr   # FDR only
    uv run python scripts/benchmark_docetl_exp.py --skip-fdr   # RTR only
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evaluate import evaluate
from src.extraction.run_docetl import run_pipeline

# ── Paths ────────────────────────────────────────────────────────────────────
BENCHMARK = "exp"
MODEL     = "gpt-4o-mini"

DATA_DIR        = REPO_ROOT / "data"
PROCESSED_DIR   = DATA_DIR / "processed" / BENCHMARK
PAPERS_JSON     = PROCESSED_DIR / "papers.json"
PRED_DIR        = DATA_DIR / "predictions" / BENCHMARK / MODEL
GT_PATH         = DATA_DIR / "benchmark" / "EXP_groundtruth.csv"

PIPELINE_RTR    = REPO_ROOT / "pipelines" / "pipeline_rtr.yaml"
PIPELINE_FDR    = REPO_ROOT / "pipelines" / "pipeline_fdr.yaml"

# DocETL intermediate checkpoint directories (one per pipeline variant).
INTER_RTR = PROCESSED_DIR / f".docetl_cache_{BENCHMARK}_{MODEL}_rtr"
INTER_FDR = PROCESSED_DIR / f".docetl_cache_{BENCHMARK}_{MODEL}_fdr"

# DocETL LLM cache (SQLite) – cleared between runs for independent samples.
try:
    from docetl.operations.utils import CACHE_DIR, LLM_CACHE_DIR, clear_cache
    _DOCETL_CACHE_DIR     = Path(CACHE_DIR)
    _DOCETL_LLM_CACHE_DIR = Path(LLM_CACHE_DIR)
except Exception:
    _DOCETL_CACHE_DIR     = Path.home() / ".cache" / "docetl" / "general"
    _DOCETL_LLM_CACHE_DIR = Path.home() / ".cache" / "docetl" / "llm"
    clear_cache = None  # will fall back to manual deletion


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clear_docetl_caches(verbose: bool = True) -> None:
    """Wipe the DocETL LLM cache and both intermediate dirs.

    IMPORTANT: diskcache keeps an open SQLite connection to LLM_CACHE_DIR as a
    module-level singleton. We must use its own clear() method (via DocETL's
    clear_cache()) rather than rmtree-ing the directory — deleting the file
    while the connection is open corrupts the handle and causes
    'no such table: Cache' on the next access.
    """
    # 1. LLM cache: use DocETL's clear_cache() which calls diskcache's c.clear()
    #    This empties all cached LLM responses while keeping the schema intact.
    if clear_cache is not None:
        try:
            clear_cache()
        except Exception:
            pass
    else:
        # Fallback: use diskcache directly if clear_cache wasn't importable
        try:
            import diskcache
            with diskcache.Cache(_DOCETL_LLM_CACHE_DIR) as c:
                c.clear()
        except Exception:
            pass

    # 2. General cache dir: safe to delete files (not an open connection)
    if _DOCETL_CACHE_DIR.exists():
        shutil.rmtree(_DOCETL_CACHE_DIR, ignore_errors=True)
    _DOCETL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Intermediate checkpoint dirs (safe to rmtree — not an open connection)
    for inter in [INTER_RTR, INTER_FDR]:
        if inter.exists():
            shutil.rmtree(inter, ignore_errors=True)

    if verbose:
        print("  [cache] LLM cache cleared, intermediate checkpoints removed.")


def _extract_metrics(metrics: dict) -> dict:
    """Pull the scalar values we care about from an evaluate() result dict."""
    pm = metrics.get("pair_micro", {})
    tm = metrics.get("triple_micro", {})
    pma = metrics.get("pair_macro", {})
    tma = metrics.get("triple_macro", {})
    return {
        "pair_precision":  pm.get("precision",  float("nan")),
        "pair_recall":     pm.get("recall",      float("nan")),
        "pair_f1":         pm.get("f1",          float("nan")),
        "triple_precision": tm.get("precision",  float("nan")),
        "triple_recall":   tm.get("recall",      float("nan")),
        "triple_f1":       tm.get("f1",          float("nan")),
        "macro_pair_f1":   pma.get("f1",          float("nan")),
        "macro_triple_f1": tma.get("f1",          float("nan")),
    }


def _mean(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def _std(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    if len(finite) < 2:
        return float("nan")
    m = _mean(finite)
    return math.sqrt(sum((x - m) ** 2 for x in finite) / (len(finite) - 1))


def _aggregate(results: list[dict]) -> dict:
    """Compute mean ± std for every metric key across runs."""
    keys = list(results[0].keys())
    agg = {}
    for k in keys:
        vals = [r[k] for r in results]
        agg[k] = {"mean": _mean(vals), "std": _std(vals), "values": vals}
    return agg


def _print_summary(label: str, agg: dict) -> None:
    n = len(next(iter(agg.values()))["values"])
    print(f"\n{'='*60}")
    print(f"  {label}  ({n} runs)")
    print(f"{'='*60}")
    fmt = "{:<22}  {:>8}  {:>8}"
    print(fmt.format("Metric", "Mean", "Std"))
    print("-" * 42)
    for k, v in agg.items():
        print(fmt.format(k, f"{v['mean']:.4f}", f"{v['std']:.4f}"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs",        type=int, default=20,
                   help="Number of independent runs (default: 20).")
    p.add_argument("--max-threads", type=int, default=21,
                   help="DocETL max_threads per pipeline run (default: 21 = one thread per EXP paper).")
    p.add_argument("--skip-rtr",    action="store_true", help="Skip RTR pipeline.")
    p.add_argument("--skip-fdr",    action="store_true", help="Skip FDR pipeline.")
    p.add_argument("--out",         type=Path, default=None,
                   help="Save full per-run JSON results to this file.")
    args = p.parse_args(argv)

    if not PAPERS_JSON.exists():
        sys.exit(f"[error] papers.json not found at {PAPERS_JSON}\n"
                 "Run the pipeline with --skip-rtr --skip-fdr --skip-datagatherer first "
                 "to preprocess the EXP papers.")
    if not GT_PATH.exists():
        sys.exit(f"[error] Ground truth not found at {GT_PATH}\n"
                 "Run: uv run python scripts/download_benchmarks.py")

    PRED_DIR.mkdir(parents=True, exist_ok=True)

    rtr_runs: list[dict] = []
    fdr_runs: list[dict] = []

    for run_idx in range(1, args.runs + 1):
        print(f"\n{'#'*60}")
        print(f"  Run {run_idx}/{args.runs}")
        print(f"{'#'*60}")

        # Clear all caches BEFORE this run so each run is independent.
        _clear_docetl_caches(verbose=True)

        # ── RTR ──────────────────────────────────────────────────────────────
        if not args.skip_rtr:
            rtr_pred = PRED_DIR / "rtr_predictions.jsonl"
            metrics_path = PRED_DIR / f"_bench_rtr_run{run_idx:02d}.json"
            print(f"\n[Run {run_idx}] DocETL RTR ...")
            try:
                run_pipeline(
                    input_path=PAPERS_JSON,
                    output_path=rtr_pred,
                    pipeline_yaml=PIPELINE_RTR,
                    model=MODEL,
                    intermediate_dir=INTER_RTR,
                    max_threads=args.max_threads,
                )
                metrics = evaluate(
                    predictions_path=rtr_pred,
                    groundtruth_path=GT_PATH,
                    output_path=metrics_path,
                    label="docetl_rtr",
                    model=MODEL,
                    benchmark=BENCHMARK,
                )
                run_metrics = _extract_metrics(metrics)
                rtr_runs.append(run_metrics)
                print(f"  pair_f1={run_metrics['pair_f1']:.4f}  "
                      f"triple_f1={run_metrics['triple_f1']:.4f}  "
                      f"macro_pair_f1={run_metrics['macro_pair_f1']:.4f}")
            except Exception:
                print(f"  [warn] RTR run {run_idx} failed:")
                traceback.print_exc()

        # ── FDR ──────────────────────────────────────────────────────────────
        if not args.skip_fdr:
            fdr_pred = PRED_DIR / "fdr_predictions.jsonl"
            metrics_path = PRED_DIR / f"_bench_fdr_run{run_idx:02d}.json"
            print(f"\n[Run {run_idx}] DocETL FDR ...")
            try:
                run_pipeline(
                    input_path=PAPERS_JSON,
                    output_path=fdr_pred,
                    pipeline_yaml=PIPELINE_FDR,
                    model=MODEL,
                    intermediate_dir=INTER_FDR,
                    max_threads=args.max_threads,
                )
                metrics = evaluate(
                    predictions_path=fdr_pred,
                    groundtruth_path=GT_PATH,
                    output_path=metrics_path,
                    label="docetl_fdr",
                    model=MODEL,
                    benchmark=BENCHMARK,
                )
                run_metrics = _extract_metrics(metrics)
                fdr_runs.append(run_metrics)
                print(f"  pair_f1={run_metrics['pair_f1']:.4f}  "
                      f"triple_f1={run_metrics['triple_f1']:.4f}  "
                      f"macro_pair_f1={run_metrics['macro_pair_f1']:.4f}")
            except Exception:
                print(f"  [warn] FDR run {run_idx} failed:")
                traceback.print_exc()

    # ── Aggregate & print ─────────────────────────────────────────────────────
    print(f"\n\n{'*'*60}")
    print(f"  FINAL AGGREGATED RESULTS  ({args.runs} runs, max_threads={args.max_threads})")
    print(f"  Benchmark: DataRef-EXP   Model: {MODEL}")
    print(f"{'*'*60}")

    all_results = {}

    if rtr_runs:
        rtr_agg = _aggregate(rtr_runs)
        all_results["DocETL RTR"] = rtr_agg
        _print_summary("DocETL RTR", rtr_agg)

    if fdr_runs:
        fdr_agg = _aggregate(fdr_runs)
        all_results["DocETL FDR"] = fdr_agg
        _print_summary("DocETL FDR", fdr_agg)

    # ── Save full results to JSON ─────────────────────────────────────────────
    out_path = args.out or REPO_ROOT / "outputs" / "docetl_exp_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": BENCHMARK,
        "model": MODEL,
        "n_runs": args.runs,
        "max_threads": args.max_threads,
        "methods": all_results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  Full results saved to: {out_path}")


if __name__ == "__main__":
    main()
