"""Print a side-by-side comparison of DocETL vs DataGatherer metrics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(p: Path) -> dict | None:
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _row(label: str, a: float | int, b: float | int, fmt: str = "{:>8}") -> str:
    return f"  {label:<22} " + fmt.format(a) + "   " + fmt.format(b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docetl", type=Path, default=REPO_ROOT / "outputs" / "metrics_docetl.json")
    ap.add_argument("--datagatherer", type=Path, default=REPO_ROOT / "outputs" / "metrics_datagatherer.json")
    ap.add_argument("--docetl-cost", type=Path, default=REPO_ROOT / "outputs" / "cost_docetl.json")
    ap.add_argument("--dg-run", type=Path, default=REPO_ROOT / "outputs" / "run_datagatherer.json")
    args = ap.parse_args()

    d = _load(args.docetl)
    g = _load(args.datagatherer)
    if not d:
        print(f"[ERROR] {args.docetl} not found. Run scripts/evaluate_docetl.py first.")
        return 1
    if not g:
        print(f"[ERROR] {args.datagatherer} not found. Run DataGatherer + evaluate first.")
        return 1

    cost = _load(args.docetl_cost) or {}
    dgrun = _load(args.dg_run) or {}

    def col(m, key, sub):
        return m.get(key, {}).get(sub, "-")

    print()
    print(f"{'METRIC':<24} {'DocETL':>10}   {'DataGatherer':>12}")
    print("-" * 52)
    for sub in ("precision", "recall", "f1"):
        print(_row(f"pair_micro.{sub}", col(d, "pair_micro", sub), col(g, "pair_micro", sub), "{:>10}"))
    for sub in ("tp", "fp", "fn"):
        print(_row(f"pair_micro.{sub}", col(d, "pair_micro", sub), col(g, "pair_micro", sub), "{:>10}"))
    print()
    for sub in ("precision", "recall", "f1"):
        print(_row(f"pair_macro.{sub}", col(d, "pair_macro", sub), col(g, "pair_macro", sub), "{:>10}"))
    print()
    for sub in ("precision", "recall", "f1"):
        print(_row(f"triple_micro.{sub}", col(d, "triple_micro", sub), col(g, "triple_micro", sub), "{:>10}"))
    print()
    print(_row("papers_with_pred",
               col(d, "coverage", "n_papers_with_prediction"),
               col(g, "coverage", "n_papers_with_prediction"), "{:>10}"))
    print(_row("total_predictions",
               col(d, "coverage", "n_total_predictions"),
               col(g, "coverage", "n_total_predictions"), "{:>10}"))
    print(_row("total_groundtruth",
               col(d, "coverage", "n_total_groundtruth"),
               col(g, "coverage", "n_total_groundtruth"), "{:>10}"))
    print()

    # Cost / runtime
    docetl_cost = (cost.get("cost") or {}).get("docetl_reported_cost_usd")
    docetl_cost = docetl_cost if docetl_cost is not None else (cost.get("cost") or {}).get("approx_cost_usd")
    dg_elapsed = dgrun.get("elapsed_sec")
    print(f"  {'docetl_cost_usd':<22} {str(docetl_cost or '-'):>10}")
    print(f"  {'datagatherer_runtime_s':<22} {'':>10}   {str(dg_elapsed or '-'):>12}")
    print()

    # Failure category counts
    df = d.get("failures", {}).get("pair", {})
    gf = g.get("failures", {}).get("pair", {})
    print("FAILURE COUNTS (pair-level)")
    print("-" * 52)
    for cat in ("missed_identifier", "hallucinated_identifier", "incomplete_identifier", "wrong_repository"):
        a = len(df.get(cat, []))
        b = len(gf.get(cat, []))
        print(f"  {cat:<24} {a:>10}   {b:>12}")
    print()

    # Verdict
    d_f1 = d["pair_micro"]["f1"]
    g_f1 = g["pair_micro"]["f1"]
    if d_f1 > g_f1:
        print(f"DocETL pair F1 is HIGHER by {d_f1 - g_f1:+.3f}")
    elif g_f1 > d_f1:
        print(f"DataGatherer pair F1 is HIGHER by {g_f1 - d_f1:+.3f}")
    else:
        print("Tie on pair F1.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
