"""Generate the final outputs/report.md from saved metrics + run summaries."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


# ── Display helpers ───────────────────────────────────────────────────────────

_BENCHMARK_DISPLAY = {"exp": "DataRef-EXP", "rev": "DataRef-REV"}
_METHOD_DISPLAY = {
    "rtr":              "DocETL RTR",
    "fdr":              "DocETL FDR",
    "datagatherer_rtr": "DG-RTR",
    "datagatherer_fdr": "DG-FDR",
}
_METHOD_ORDER = ["rtr", "fdr", "datagatherer_rtr", "datagatherer_fdr"]
_BENCHMARK_ORDER = ["exp", "rev"]

# Columns for the unified matrix table: (metrics_key, sub_key, header)
_MATRIX_COLS: list[tuple[str, str, str]] = [
    ("pair_micro",   "precision", "Pair Precision"),
    ("pair_micro",   "recall",    "Pair Recall"),
    ("pair_micro",   "f1",        "Pair F1"),
    ("triple_micro", "precision", "Triple Precision"),
    ("triple_micro", "recall",    "Triple Recall"),
    ("triple_micro", "f1",        "Triple F1"),
]


def _maybe_load(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _display_model(model: str) -> str:
    """Strip LiteLLM provider prefix for compact display.

    'gemini/gemini-2.5-flash' -> 'gemini-2.5-flash'
    'openai/gpt-4o-mini'      -> 'gpt-4o-mini'
    'gpt-4o-mini'             -> 'gpt-4o-mini'
    """
    return model.split("/", 1)[1] if "/" in model else model


def _parse_metrics_filename(stem: str) -> tuple[str, str, str] | None:
    """Parse '{benchmark}_{safe_model}_{method}' -> (benchmark, safe_model, method) or None."""
    # Check compound suffixes first so 'datagatherer_rtr'/'datagatherer_fdr' beat plain 'rtr'/'fdr'
    for method in ("datagatherer_rtr", "datagatherer_fdr", "fdr", "rtr"):
        if stem.endswith(f"_{method}"):
            prefix = stem[: -len(f"_{method}")]
            for bm in _BENCHMARK_ORDER:
                if prefix.startswith(f"{bm}_"):
                    return bm, prefix[len(f"{bm}_"):], method
    return None


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return "-" if v is None else str(v)


# ── Matrix table ──────────────────────────────────────────────────────────────

def _build_matrix_table(all_metrics: list[dict[str, Any]]) -> str:
    """Build the Dataset × Model × Method comparison table with bold best-per-group values."""
    if not all_metrics:
        return "_No metrics available yet._\n"

    # Group by (benchmark, model)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for m in all_metrics:
        groups[(m.get("benchmark", ""), m.get("model", ""))].append(m)

    def _group_key(k: tuple[str, str]) -> tuple[int, str]:
        bm, model = k
        return (_BENCHMARK_ORDER.index(bm) if bm in _BENCHMARK_ORDER else 99, model)

    headers = ["Dataset", "Model", "Method"] + [hdr for _, _, hdr in _MATRIX_COLS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]

    for gkey in sorted(groups.keys(), key=_group_key):
        bm, model = gkey
        group = sorted(
            groups[gkey],
            key=lambda m: (
                _METHOD_ORDER.index(m.get("label", ""))
                if m.get("label") in _METHOD_ORDER
                else 99
            ),
        )

        # Find best value per metric column within this (dataset, model) group for bolding
        best: dict[tuple[str, str], float] = {}
        for sect, key, _ in _MATRIX_COLS:
            nums = [
                v for m in group
                if isinstance(v := m.get(sect, {}).get(key), (int, float))
            ]
            if nums:
                best[(sect, key)] = max(nums)

        dataset_display = _BENCHMARK_DISPLAY.get(bm, bm.upper())
        model_display = _display_model(model)

        for row in group:
            method_key = row.get("label", "")
            method_display = _METHOD_DISPLAY.get(method_key, method_key.upper())
            cells = [dataset_display, model_display, method_display]
            for sect, key, _ in _MATRIX_COLS:
                v = row.get(sect, {}).get(key)
                s = _fmt(v)
                if isinstance(v, (int, float)) and best.get((sect, key)) == v:
                    s = f"**{s}**"
                cells.append(s)
            lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


# ── Per-run detail sections ───────────────────────────────────────────────────

def _detail_table(metrics: dict[str, Any] | None, label: str) -> str:
    if not metrics:
        return f"_No metrics available for {label}._\n"
    pm = metrics.get("pair_micro", {})
    tm = metrics.get("triple_micro", {})
    lines = [
        f"#### {label}",
        "",
        "| Metric | Pair (paper, dataset_id) | Triple (paper, dataset_id, repository) |",
        "|---|---|---|",
    ]
    for key in ("precision", "recall", "f1", "tp", "fp", "fn"):
        lines.append(f"| {key} | {pm.get(key, '-')} | {tm.get(key, '-')} |")
    lines.append("")
    macro_p = metrics.get("pair_macro", {})
    macro_t = metrics.get("triple_macro", {})
    lines.append(
        f"Macro-average pair F1: **{macro_p.get('f1', '-')}** "
        f"(over {macro_p.get('n_papers', 0)} papers); "
        f"triple F1: **{macro_t.get('f1', '-')}**."
    )
    cov = metrics.get("coverage", {})
    if cov:
        cov_lines = [
            "",
            "Coverage:",
            f"- Papers with at least one prediction: {cov.get('n_papers_with_prediction', 0)}",
            f"- Papers with at least one ground-truth ref: {cov.get('n_papers_with_groundtruth', 0)}",
            f"- Empty outputs: {cov.get('n_empty_outputs', 0)}",
        ]
        if "n_real_predictions" in cov:
            cov_lines.append(f"- Real predictions (non-N/A): {cov['n_real_predictions']}")
            cov_lines.append(f"- N/A placeholder rows: {cov['n_na_predictions']}")
        cov_lines.append(f"- Total predictions: {cov.get('n_total_predictions', 0)}")
        cov_lines.append(f"- Total ground-truth refs: {cov.get('n_total_groundtruth', 0)}")
        lines.extend(cov_lines)
    lines.append("")
    return "\n".join(lines)


def _failure_examples(metrics: dict[str, Any] | None, k: int = 5) -> str:
    if not metrics:
        return ""
    fails = metrics.get("failures", {}).get("pair", {})
    out: list[str] = ["", "##### Failure examples (pair-level)", ""]
    for cat in ("missed_identifier", "hallucinated_identifier", "incomplete_identifier"):
        items = fails.get(cat, [])[:k]
        if not items:
            continue
        out.append(f"- **{cat}** ({len(fails.get(cat, []))} total):")
        for t in items:
            out.append(f"  - {t}")
    return "\n".join(out) + "\n"


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_report(
    metrics_dir: Path | None = None,
    run_summaries: list[dict[str, Any]] | None = None,
    output_path: Path = Path("outputs/report.md"),
    # Legacy single-path params kept for backward compatibility
    rtr_metrics_path: Path | None = None,
    fdr_metrics_path: Path | None = None,
    datagatherer_metrics_path: Path | None = None,
    rtr_run_summary: dict[str, Any] | None = None,
    fdr_run_summary: dict[str, Any] | None = None,
    datagatherer_run_summary: dict[str, Any] | None = None,
    docetl_metrics_path: Path | None = None,
    docetl_run_summary: dict[str, Any] | None = None,
) -> Path:
    # Support legacy single-strategy callers
    if docetl_metrics_path is not None:
        rtr_metrics_path = docetl_metrics_path
    if docetl_run_summary is not None:
        rtr_run_summary = docetl_run_summary

    # ── Collect all metrics ───────────────────────────────────────────────────
    all_metrics: list[dict[str, Any]] = []

    if metrics_dir is not None:
        for p in sorted(Path(metrics_dir).glob("*.json")):
            m = _maybe_load(p)
            if m is None:
                continue
            # Back-fill benchmark/model from filename if the JSON predates those fields
            if not m.get("benchmark") or not m.get("model"):
                parsed = _parse_metrics_filename(p.stem)
                if parsed:
                    bm, safe_model, method = parsed
                    m.setdefault("benchmark", bm)
                    m.setdefault("label", method)
                    if not m.get("model"):
                        m["model"] = safe_model  # best we can do without the original string
            all_metrics.append(m)
    else:
        # Legacy fallback: explicit per-path args
        for path, label in [
            (rtr_metrics_path, "rtr"),
            (fdr_metrics_path, "fdr"),
            (datagatherer_metrics_path, "datagatherer"),
        ]:
            m = _maybe_load(path)
            if m:
                m.setdefault("label", label)
                m.setdefault("model", "unknown")
                m.setdefault("benchmark", "exp")
                all_metrics.append(m)

    # ── Build report sections ─────────────────────────────────────────────────
    parts: list[str] = []

    parts.append("# Dataset Reference Extraction — Results\n")

    # ── Section 1: Unified comparison table ──────────────────────────────────
    parts.append(
        "## 1. Unified Comparison\n\n"
        "Bold = best within each Dataset × Model group.\n"
        "Pair = (paper_id, dataset_identifier); Triple = (paper_id, dataset_identifier, repository).\n"
    )
    parts.append(_build_matrix_table(all_metrics))

    # ── Section 2: Per-run detailed metrics ───────────────────────────────────
    parts.append("## 2. Detailed Metrics per Run\n")

    # Group by (benchmark, model) then iterate methods in order
    by_bm_model: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for m in all_metrics:
        bm = m.get("benchmark", "")
        model = m.get("model", "")
        method = m.get("label", "")
        by_bm_model[(bm, model)][method] = m

    def _bm_model_key(k: tuple[str, str]) -> tuple[int, str]:
        bm, model = k
        return (_BENCHMARK_ORDER.index(bm) if bm in _BENCHMARK_ORDER else 99, model)

    for (bm, model) in sorted(by_bm_model.keys(), key=_bm_model_key):
        ds_display = _BENCHMARK_DISPLAY.get(bm, bm.upper())
        m_display = _display_model(model)
        parts.append(f"### {ds_display} | {m_display}\n")
        method_map = by_bm_model[(bm, model)]
        for method in _METHOD_ORDER:
            met = method_map.get(method)
            section_label = f"{ds_display} — {m_display} — {_METHOD_DISPLAY.get(method, method.upper())}"
            parts.append(_detail_table(met, section_label))
            parts.append(_failure_examples(met))

    # ── Section 8: Cost / runtime ─────────────────────────────────────────────
    cost_lines: list[str] = []
    if run_summaries:
        for entry in run_summaries:
            bm = entry.get("benchmark", "")
            model = entry.get("model", "")
            prefix = f"{_BENCHMARK_DISPLAY.get(bm, bm.upper())} | {_display_model(model)}"
            for method_key, label_key in [
                    ("rtr",    "rtr"),
                    ("fdr",    "fdr"),
                    ("dg_rtr", "datagatherer_rtr"),
                    ("dg_fdr", "datagatherer_fdr"),
                ]:
                s = entry.get(method_key)
                if s:
                    method_label = _METHOD_DISPLAY.get(label_key, label_key.upper())
                    cost_lines.append(
                        f"- {prefix} | {method_label}: {json.dumps(s.get('cost', s))}"
                    )
    else:
        # Legacy fallback
        for label, summary in [
            ("RTR", rtr_run_summary),
            ("FDR", fdr_run_summary),
            ("DataGatherer", datagatherer_run_summary),
        ]:
            if summary:
                cost_lines.append(f"- {label}: {json.dumps(summary.get('cost', summary))}")

    if cost_lines:
        parts.append("\n## 3. Cost / Runtime\n\n" + "\n".join(cost_lines) + "\n")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path
