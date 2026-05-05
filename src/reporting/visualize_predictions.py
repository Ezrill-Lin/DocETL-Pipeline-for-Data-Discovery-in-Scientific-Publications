"""Render row-by-row prediction vs. ground-truth HTML tables per model.

For each model directory under ``data/predictions/{benchmark}/``, scan every
``*_predictions.jsonl`` file and emit one self-contained HTML file at
``outputs/visualizations/{benchmark}_{model}.html``.

Per method, two tables are rendered:

1. **Predictions** — one row per predicted (paper, dataset_identifier).
   Columns: paper, predicted (id / repo / url), match status, ground-truth
   (id / repo / url). Status is TP (matches GT id), REPO-MISMATCH (id matches
   but repo differs), or FP (no matching GT id — GT columns filled with N/A).

2. **Missed (FN)** — ground-truth rows with no matching prediction for the
   same paper.

Usage:
    python -m src.reporting.visualize_predictions [--benchmark exp]
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.load_groundtruth import load_groundtruth, normalize_groundtruth_row
from src.evaluation.match_records import build_paper_index, resolve_paper
from src.extraction.registry import is_na_identifier, synonym_group
from src.extraction.url_builder import normalize_identifier


_METHOD_ORDER = ["rtr", "fdr", "datagatherer_rtr", "datagatherer_fdr"]
_METHOD_DISPLAY = {
    "rtr": "DocETL RTR",
    "fdr": "DocETL FDR",
    "datagatherer_rtr": "DG-RTR",
    "datagatherer_fdr": "DG-FDR",
}


def _method_from_filename(stem: str) -> str | None:
    """Map a prediction filename stem to a method key."""
    base = stem.replace("_predictions", "")
    if base == "datagatherer":  # legacy filename for the RTR variant
        return "datagatherer_rtr"
    return base if base in _METHOD_ORDER else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _esc(v: Any) -> str:
    if v is None:
        return ""
    return html.escape(str(v))


def _link(url: str) -> str:
    if not url:
        return "<span class='na'>N/A</span>"
    safe = html.escape(url, quote=True)
    return f"<a href='{safe}' target='_blank'>{html.escape(url)}</a>"


def _classify(
    pred_repo: str,
    matched_gt: list[dict[str, Any]],
) -> tuple[str, dict[str, Any] | None]:
    """Decide TP / REPO-MISMATCH / FP and pick a single GT row to show."""
    if not matched_gt:
        return "FP", None
    pred_group = synonym_group(pred_repo)
    # Prefer a GT row whose repo synonym-matches the prediction
    for g in matched_gt:
        if synonym_group(g.get("repository", "")) == pred_group:
            return "TP", g
    return "REPO-MISMATCH", matched_gt[0]


# ── HTML rendering ───────────────────────────────────────────────────────────

_CSS = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 1600px; margin: 24px auto; padding: 0 16px;
         color: #222; background: #fafafa; }
  h1 { font-size: 22px; }
  h2 { font-size: 18px; margin-top: 32px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
  h3 { font-size: 14px; margin: 0 0 6px 0; color: #555; font-weight: 600; }
  h4 { font-size: 14px; margin: 18px 0 6px 0; color: #333;
       background: #eef1f5; padding: 6px 10px; border-left: 3px solid #1a73e8; }
  table { border-collapse: collapse; width: 100%; font-size: 12px;
          background: white; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
  th, td { border: 1px solid #e3e3e3; padding: 5px 7px; text-align: left;
           vertical-align: top; word-break: break-word; }
  th { background: #f0f0f0; }
  tr.tp   { background: #e8f6ec; }
  tr.fp   { background: #fbe9e9; }
  tr.repo { background: #fff5d6; }
  tr.fn   { background: #fef0e0; }
  .badge { display: inline-block; padding: 1px 5px; border-radius: 3px;
           font-weight: 600; font-size: 10.5px; }
  .b-tp   { background: #2e7d32; color: white; }
  .b-fp   { background: #c62828; color: white; }
  .b-repo { background: #b88600; color: white; }
  .b-fn   { background: #d35400; color: white; }
  .b-na   { background: #888;    color: white; }
  .na    { color: #999; font-style: italic; }
  a      { color: #1a73e8; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .summary { margin: 8px 0 16px; font-size: 13px; color: #555; }
  .summary span { display: inline-block; margin-right: 14px; }
  /* Paper block: two side-by-side dataframes */
  .paper-block { display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
                 margin-bottom: 18px; }
  .half { min-width: 0; }
  .half .empty { padding: 8px 10px; color: #999; font-style: italic;
                 background: white; border: 1px solid #e3e3e3; }
</style>
"""


def _render_pred_row(idx: int, pred: dict[str, Any], status: str) -> str:
    cls = {"TP": "tp", "FP": "fp", "REPO-MISMATCH": "repo"}[status]
    badge = {"TP": "b-tp", "FP": "b-fp", "REPO-MISMATCH": "b-repo"}[status]
    return (
        f"<tr class='{cls}'>"
        f"<td>{idx}</td>"
        f"<td>{_esc(pred.get('dataset_identifier', ''))}</td>"
        f"<td>{_esc(pred.get('repository', ''))}</td>"
        f"<td>{_link(pred.get('url', ''))}</td>"
        f"<td><span class='badge {badge}'>{status}</span></td>"
        f"</tr>"
    )


def _render_gt_row(idx: int, gt: dict[str, Any], status: str) -> str:
    cls = {"TP": "tp", "REPO-MISMATCH": "repo", "FN": "fn"}[status]
    badge = {"TP": "b-tp", "REPO-MISMATCH": "b-repo", "FN": "b-fn"}[status]
    return (
        f"<tr class='{cls}'>"
        f"<td>{idx}</td>"
        f"<td>{_esc(gt.get('dataset_identifier', ''))}</td>"
        f"<td>{_esc(gt.get('repository', ''))}</td>"
        f"<td>{_link(gt.get('url', ''))}</td>"
        f"<td><span class='badge {badge}'>{status}</span></td>"
        f"</tr>"
    )


def _half_table(title: str, header_html: str, body_rows: list[str]) -> str:
    if body_rows:
        body = "".join(body_rows)
        table = (
            f"<table><thead>{header_html}</thead><tbody>{body}</tbody></table>"
        )
    else:
        table = "<div class='empty'>(none)</div>"
    return f"<div class='half'><h3>{html.escape(title)}</h3>{table}</div>"


def render_model(
    model_dir: Path,
    groundtruth: list[dict[str, Any]],
    output_path: Path,
    benchmark: str,
) -> None:
    """Render an HTML file for one model directory."""
    model_label = model_dir.name

    # Discover prediction files & sort by canonical method order
    found: dict[str, Path] = {}
    for p in model_dir.glob("*_predictions.jsonl"):
        method = _method_from_filename(p.stem)
        if method:
            found[method] = p

    parts: list[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(benchmark.upper())} — {html.escape(model_label)} predictions</title>",
        _CSS,
        "</head><body>",
        f"<h1>{html.escape(benchmark.upper())} — {html.escape(model_label)}</h1>",
        f"<div class='summary'>Predictions vs. ground truth, row-by-row. "
        f"Matching is at (paper, normalized_dataset_id) level.</div>",
    ]

    methods_present = [m for m in _METHOD_ORDER if m in found]
    if not methods_present:
        parts.append("<p><em>No prediction files found in this directory.</em></p>")

    pred_header = (
        "<tr><th>#</th><th>ID</th><th>Repo</th><th>URL</th><th>Status</th></tr>"
    )
    gt_header = pred_header  # same columns

    for method in methods_present:
        preds_path = found[method]
        raw_preds = _read_jsonl(preds_path)
        norm_preds = [normalize_groundtruth_row(r) for r in raw_preds]
        pred_index = build_paper_index(norm_preds)
        gt_index = build_paper_index(groundtruth)

        # Group predictions by canonical paper key (preserving raw values for display)
        preds_by_paper: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
        for raw, norm in zip(raw_preds, norm_preds):
            if is_na_identifier(norm.get("dataset_identifier", "")):
                continue
            ckey = resolve_paper(norm, [pred_index, gt_index])
            preds_by_paper.setdefault(ckey, []).append((raw, norm))

        # Group groundtruth by canonical paper key (skipping N/A sentinel rows)
        gts_by_paper: dict[str, list[dict[str, Any]]] = {}
        for g in groundtruth:
            if is_na_identifier(g.get("dataset_identifier", "")):
                continue
            ckey = resolve_paper(g, [gt_index, pred_index])
            gts_by_paper.setdefault(ckey, []).append(g)

        all_papers = sorted(set(preds_by_paper) | set(gts_by_paper))

        # Tally counts across all papers for the per-method summary banner
        n_tp = n_fp = n_repo = n_fn = 0
        paper_blocks: list[str] = []

        for ckey in all_papers:
            preds_for_paper = preds_by_paper.get(ckey, [])
            gts_for_paper = gts_by_paper.get(ckey, [])

            # Pick a display label for the paper from whichever side has data
            label_row = (preds_for_paper[0][0] if preds_for_paper
                         else gts_for_paper[0])
            paper_label = (label_row.get("paper_id")
                           or label_row.get("pmcid")
                           or ckey)

            # Build a {normalized_id: list[gt_row]} index for this paper to
            # classify each prediction and detect unmatched GT rows.
            gt_by_id: dict[str, list[dict[str, Any]]] = {}
            for g in gts_for_paper:
                ds_norm = normalize_identifier(g.get("dataset_identifier", "")).lower()
                if ds_norm:
                    gt_by_id.setdefault(ds_norm, []).append(g)

            matched_gt_ids: set[int] = set()
            pred_rows: list[str] = []
            for i, (raw, norm) in enumerate(preds_for_paper, start=1):
                ds_norm = normalize_identifier(norm.get("dataset_identifier", "")).lower()
                matched_gt = gt_by_id.get(ds_norm, [])
                status, picked = _classify(raw.get("repository", ""), matched_gt)
                if picked is not None:
                    matched_gt_ids.add(id(picked))
                if status == "TP":
                    n_tp += 1
                elif status == "REPO-MISMATCH":
                    n_repo += 1
                else:
                    n_fp += 1
                pred_rows.append(_render_pred_row(i, raw, status))

            # Classify each GT row: TP / REPO-MISMATCH (matched) or FN (unmatched)
            gt_rows: list[str] = []
            for j, g in enumerate(gts_for_paper, start=1):
                ds_norm = normalize_identifier(g.get("dataset_identifier", "")).lower()
                # A GT row's status mirrors what a matching prediction (if any) got.
                status = "FN"
                if id(g) in matched_gt_ids:
                    status = "TP"
                else:
                    # Was *some* prediction's id this GT's id, but with a different repo?
                    # We treat that as REPO-MISMATCH on the GT side too.
                    for raw, norm in preds_for_paper:
                        if normalize_identifier(norm.get("dataset_identifier", "")).lower() == ds_norm:
                            status = "REPO-MISMATCH"
                            break
                if status == "FN":
                    n_fn += 1
                gt_rows.append(_render_gt_row(j, g, status))

            paper_blocks.append(
                f"<h4>Paper: {html.escape(str(paper_label))} "
                f"<span style='font-weight:400; color:#666;'>"
                f"({len(preds_for_paper)} pred / {len(gts_for_paper)} gt)</span></h4>"
                f"<div class='paper-block'>"
                f"{_half_table('Predictions', pred_header, pred_rows)}"
                f"{_half_table('Ground Truth', gt_header, gt_rows)}"
                f"</div>"
            )

        parts.append(f"<h2>{html.escape(_METHOD_DISPLAY[method])}</h2>")
        parts.append(
            f"<div class='summary'>"
            f"<span><span class='badge b-tp'>TP</span> {n_tp}</span>"
            f"<span><span class='badge b-repo'>REPO-MISMATCH</span> {n_repo}</span>"
            f"<span><span class='badge b-fp'>FP</span> {n_fp}</span>"
            f"<span><span class='badge b-fn'>FN</span> {n_fn}</span>"
            f"<span>papers: {len(all_papers)}</span>"
            f"<span>file: <code>{html.escape(preds_path.name)}</code></span>"
            f"</div>"
        )
        parts.extend(paper_blocks or ["<p><em>No paper-level data.</em></p>"])

    parts.append("</body></html>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default="exp", choices=["exp", "rev"])
    p.add_argument("--predictions-root", type=Path,
                   default=REPO_ROOT / "data" / "predictions")
    p.add_argument("--groundtruth", type=Path, default=None,
                   help="Override path to the benchmark CSV.")
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "outputs" / "visualizations")
    args = p.parse_args(argv)

    gt_path = args.groundtruth or (
        REPO_ROOT / "data" / "benchmark" /
        ("EXP_groundtruth.csv" if args.benchmark == "exp" else "REV_sample_groundtruth.csv")
    )
    if not gt_path.exists():
        print(f"ERROR: ground truth not found at {gt_path}", file=sys.stderr)
        return 1

    groundtruth = load_groundtruth(gt_path)
    print(f"Loaded {len(groundtruth)} groundtruth rows from {gt_path.name}")

    bench_dir = args.predictions_root / args.benchmark
    if not bench_dir.exists():
        print(f"ERROR: no predictions directory at {bench_dir}", file=sys.stderr)
        return 1

    model_dirs = [d for d in sorted(bench_dir.iterdir()) if d.is_dir()]
    if not model_dirs:
        print(f"ERROR: no model subdirectories under {bench_dir}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for model_dir in model_dirs:
        out_path = args.out_dir / f"{args.benchmark}_{model_dir.name}.html"
        render_model(model_dir, groundtruth, out_path, args.benchmark)
        print(f"  → {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
