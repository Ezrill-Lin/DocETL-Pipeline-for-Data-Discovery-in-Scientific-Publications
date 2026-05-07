"""Preprocess raw REV sample papers, run DataGatherer strategy/ies, and evaluate.

Usage
-----
    # 100-paper test, RTR only (default):
    uv run python scripts/run_rev_sample_dg_rtr.py --model gpt-4o-mini

    # 5-paper smoke test, both RTR and FDR:
    uv run python scripts/run_rev_sample_dg_rtr.py --n 5 --strategies rtr,fdr --model gpt-4o-mini

Prerequisites (run once before this script)
-------------------------------------------
1. Create an N-paper sample groundtruth:
       python3 -c "
       import pandas as pd, random; random.seed(42)
       N = 5  # or 100
       gt = pd.read_csv('data/benchmark/REV_sample_groundtruth.csv')
       links = gt['citing_publication_link'].dropna().unique().tolist()
       sampled = random.sample(links, N)
       gt[gt['citing_publication_link'].isin(sampled)].to_csv(
           f'data/benchmark/REV_{N}_groundtruth.csv', index=False)
       "
2. Fetch those papers:
       python scripts/fetch_rev_papers.py \
           --gt data/benchmark/REV_<N>_groundtruth.csv \
           --out-dir data/raw/rev_<N>

Outputs (default, varies with --n)
-----------------------------------
  Preprocessed papers : data/processed/rev_<N>/papers.json
  RTR predictions     : data/predictions/rev_<N>/datagatherer_rtr_predictions.jsonl
  FDR predictions     : data/predictions/rev_<N>/datagatherer_fdr_predictions.jsonl
  RTR metrics         : outputs/metrics/rev_<N>_datagatherer_rtr.json
  FDR metrics         : outputs/metrics/rev_<N>_datagatherer_fdr.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.baselines.run_datagatherer import run_datagatherer
from src.evaluation.evaluate import evaluate
from src.preprocess.build_docetl_input import build_json_array

STRATEGY_MAP = {
    "rtr": "retrieve",
    "fdr": "full",
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=100,
                   help="Sample size used when building default paths (default: 100). "
                        "Does NOT perform sampling — run the prerequisite step first.")
    p.add_argument("--strategies", default="rtr",
                   help="Comma-separated DataGatherer strategies to run: rtr, fdr, or rtr,fdr "
                        "(default: rtr)")
    p.add_argument("--raw-dir", type=Path, default=None,
                   help="Directory of fetched raw papers (default: data/raw/rev_<N>)")
    p.add_argument("--groundtruth", type=Path, default=None,
                   help="Ground-truth CSV (default: data/benchmark/REV_<N>_groundtruth.csv)")
    p.add_argument("--papers-json", type=Path, default=None,
                   help="Where to write / reuse the preprocessed papers JSON "
                        "(default: data/processed/rev_<N>/papers.json)")
    p.add_argument("--pred-dir", type=Path, default=None,
                   help="Directory for prediction JSONL files "
                        "(default: data/predictions/rev_<N>)")
    p.add_argument("--metrics-dir", type=Path, default=None,
                   help="Directory for metrics JSON files "
                        "(default: outputs/metrics)")
    p.add_argument("--model", default="gpt-4o-mini",
                   help="LLM model name passed to DataGatherer (default: gpt-4o-mini)")
    p.add_argument("--skip-preprocess", action="store_true",
                   help="Skip preprocessing; reuse existing --papers-json file")
    args = p.parse_args(argv)

    n = args.n

    # ── Resolve default paths from --n ─────────────────────────────────────
    raw_dir     = args.raw_dir     or REPO_ROOT / f"data/raw/rev_{n}"
    gt_path     = args.groundtruth or REPO_ROOT / f"data/benchmark/REV_{n}_groundtruth.csv"
    papers_json = args.papers_json or REPO_ROOT / f"data/processed/rev_{n}/papers.json"
    pred_dir    = args.pred_dir    or REPO_ROOT / f"data/predictions/rev_{n}"
    metrics_dir = args.metrics_dir or REPO_ROOT / "outputs/metrics"

    # ── Parse strategy list ─────────────────────────────────────────────────
    requested = [s.strip().lower() for s in args.strategies.split(",")]
    unknown = [s for s in requested if s not in STRATEGY_MAP]
    if unknown:
        print(f"ERROR: Unknown strategies: {unknown}. Choose from: {list(STRATEGY_MAP)}")
        return 1

    # ── Stage 1: Preprocess ────────────────────────────────────────────────
    if args.skip_preprocess:
        if not papers_json.exists():
            print(f"ERROR: --skip-preprocess set but {papers_json} does not exist.")
            return 1
        print(f"[stage 1] Reusing existing {papers_json}")
    else:
        if not raw_dir.exists() or not any(raw_dir.rglob("*.*")):
            print(f"ERROR: No papers found in {raw_dir}. Run fetch_rev_papers.py first.")
            return 1
        papers_json.parent.mkdir(parents=True, exist_ok=True)
        n_papers = build_json_array(raw_dir, papers_json)
        print(f"[stage 1] Preprocessed {n_papers} papers → {papers_json}")

    # ── Stages 2+: DataGatherer per strategy ──────────────────────────────
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for i, strategy_key in enumerate(requested, start=2):
        dg_strategy = STRATEGY_MAP[strategy_key]
        pred_path   = pred_dir / f"datagatherer_{strategy_key}_predictions.jsonl"
        metric_path = metrics_dir / f"rev_{n}_datagatherer_{strategy_key}.json"

        print(f"\n[stage {i}] DataGatherer {strategy_key.upper()} "
              f"(strategy={dg_strategy}, model={args.model}) …")
        summary = run_datagatherer(
            papers_json,
            pred_path,
            strategy=dg_strategy,
            llm_name=args.model,
        )
        print(json.dumps(summary, indent=2))

        if not gt_path.exists():
            print(f"  [skip eval] Ground truth not found at {gt_path}")
            continue

        print(f"  Evaluating against {gt_path.name} …")
        evaluate(
            pred_path,
            gt_path,
            metric_path,
            label=f"datagatherer_{strategy_key}",
            model=args.model,
            benchmark=f"rev_{n}",
        )
        print(f"  → Metrics saved to {metric_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
