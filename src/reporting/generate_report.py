"""Generate the final outputs/report.md from saved metrics + run summaries."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _maybe_load(path: Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _metrics_table(metrics: dict[str, Any] | None, label: str) -> str:
    if not metrics:
        return f"_No metrics available for {label}._\n"
    lines = [
        f"#### {label}",
        "",
        "| Metric | Pair (paper, dataset_id) | Triple (paper, dataset_id, repository) |",
        "|---|---|---|",
    ]
    pm = metrics.get("pair_micro", {})
    tm = metrics.get("triple_micro", {})
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
        lines.append("")
        lines.append("Coverage:")
        lines.append(f"- Papers with at least one prediction: {cov.get('n_papers_with_prediction', 0)}")
        lines.append(f"- Papers with at least one ground-truth ref: {cov.get('n_papers_with_groundtruth', 0)}")
        lines.append(f"- Empty outputs: {cov.get('n_empty_outputs', 0)}")
        lines.append(f"- Total predictions: {cov.get('n_total_predictions', 0)}")
        lines.append(f"- Total ground-truth refs: {cov.get('n_total_groundtruth', 0)}")
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


def generate_report(
    rtr_metrics_path: Path,
    fdr_metrics_path: Path,
    datagatherer_metrics_path: Path | None,
    rtr_run_summary: dict[str, Any] | None,
    fdr_run_summary: dict[str, Any] | None,
    datagatherer_run_summary: dict[str, Any] | None,
    output_path: Path,
    # Legacy aliases so callers that pass docetl_metrics_path still work
    docetl_metrics_path: Path | None = None,
    docetl_run_summary: dict[str, Any] | None = None,
) -> Path:
    # Support legacy single-strategy callers
    if docetl_metrics_path is not None:
        rtr_metrics_path = docetl_metrics_path
    if docetl_run_summary is not None:
        rtr_run_summary = docetl_run_summary

    rtr_m = _maybe_load(rtr_metrics_path)
    fdr_m = _maybe_load(fdr_metrics_path)
    dg_m = _maybe_load(datagatherer_metrics_path) if datagatherer_metrics_path else None

    parts: list[str] = []
    parts.append("# Dataset Reference Extraction — RTR vs FDR vs DataGatherer\n")
    parts.append(
        "## 1. Project overview\n"
        "This project builds a DocETL pipeline that extracts dataset references "
        "from scientific papers and evaluates it on the DataRef-EXP / DataRef-REV "
        "benchmarks released by the DataGatherer authors. DataGatherer is run as "
        "an optional baseline.\n"
    )
    parts.append(
        "## 2. Pipeline architecture\n"
        "Inputs (HTML/JATS XML/PDF) are converted to structured paper records by "
        "`src/preprocess/`. Each record carries the section tree, the abstract, "
        "and a `candidate_passages` field — pre-selected passages where dataset "
        "references most often appear (data availability, methods, supplementary, "
        "figure captions, plus any paragraph mentioning a known repository or an "
        "accession-shaped token).\n\n"
        "DocETL Stage 1 is a single `map` operation that asks the LLM to return a "
        "list of `{dataset_identifier, repository, evidence, confidence, notes}` "
        "objects. The prompt enumerates known repositories and identifier "
        "patterns and forbids URL invention.\n\n"
        "DocETL Stage 2 is **deterministic Python** in "
        "`src/extraction/url_builder.py`: identifier normalization → repository "
        "alias resolution → URL templating from `config/repositories.yaml`. If "
        "the URL pattern is unknown, the URL field is left empty rather than "
        "fabricated.\n"
    )
    parts.append(
        "## 3. Why hybrid section retrieval + LLM extraction\n"
        "Pure heading-based extraction misses datasets cited inline (\"deposited "
        "into PRIDE under PXD012345\" buried in Methods) and figure captions. "
        "Pure LLM-on-full-text is wasteful — most papers' bodies are irrelevant "
        "to data availability. We retrieve passages by both heading match and a "
        "permissive regex over repository names and accession-shaped tokens; the "
        "LLM then resolves ambiguity (e.g. distinguishing a primary deposit from "
        "a re-used public dataset) on a much smaller prompt.\n"
    )
    parts.append(
        "## 4. Output schema\n"
        "Each predicted row is JSONL with fields: `paper_id`, `paper_doi`, "
        "`pmcid`, `pmid`, `dataset_identifier`, `repository`, `url`, `evidence`, "
        "`confidence`, `notes`.\n"
    )
    parts.append(
        "## 5. Evaluation dataset\n"
        "Ground truth comes from DataRef-EXP and/or DataRef-REV "
        "(https://doi.org/10.5281/zenodo.15549086). Loaders in "
        "`src/evaluation/load_groundtruth.py` accept CSV/TSV/JSON/JSONL and try a "
        "list of column-name candidates per field. Identifier and repository "
        "values are normalized identically for predictions and ground truth.\n"
    )

    parts.append("## 6. Metrics — DocETL RTR (Retrieve-Then-Read)\n")
    parts.append(_metrics_table(rtr_m, "DocETL RTR"))
    parts.append(_failure_examples(rtr_m))

    parts.append("## 7. Metrics — DocETL FDR (Full-Document Read)\n")
    parts.append(_metrics_table(fdr_m, "DocETL FDR"))
    parts.append(_failure_examples(fdr_m))

    parts.append("## 8. Metrics — DataGatherer (baseline)\n")
    if dg_m is None:
        parts.append("_DataGatherer baseline metrics not available._\n")
    else:
        parts.append(_metrics_table(dg_m, "DataGatherer"))
        parts.append(_failure_examples(dg_m))

    # Three-way comparison table
    parts.append("## 9. Comparison\n")
    rows_available = [m for m in [(rtr_m, "DocETL RTR"), (fdr_m, "DocETL FDR"), (dg_m, "DataGatherer")] if m[0]]
    if len(rows_available) >= 2:
        header = "| Aspect | " + " | ".join(label for _, label in rows_available) + " |"
        sep    = "|---|" + "---|" * len(rows_available)
        def _val(m, *keys):
            v = m
            for k in keys:
                v = (v or {}).get(k, "-")
            return v
        table_lines = [header, sep]
        for aspect, *keys in [
            ("Pair precision",  "pair_micro",   "precision"),
            ("Pair recall",     "pair_micro",   "recall"),
            ("Pair F1 (micro)", "pair_micro",   "f1"),
            ("Triple F1 (micro)", "triple_micro", "f1"),
            ("Pair F1 (macro)", "pair_macro",   "f1"),
            ("Papers covered",  "coverage",     "n_papers_with_prediction"),
            ("Total predictions", "coverage",   "n_total_predictions"),
        ]:
            vals = " | ".join(str(_val(m, *keys)) for m, _ in rows_available)
            table_lines.append(f"| {aspect} | {vals} |")
        parts.append("\n".join(table_lines) + "\n")
    else:
        parts.append("Side-by-side metrics will appear here once both runs are completed.\n")

    cost_lines = []
    for label, summary in [("RTR", rtr_run_summary), ("FDR", fdr_run_summary), ("DataGatherer", datagatherer_run_summary)]:
        if summary:
            cost_lines.append(f"- {label}: {json.dumps(summary.get('cost', summary))}")
    if cost_lines:
        parts.append("\n**Cost / runtime:**\n" + "\n".join(cost_lines) + "\n")

    parts.append(
        "\n**Engineering effort:** both DocETL strategies are defined in YAML files "
        "plus deterministic post-processing. RTR uses pre-filtered candidate passages "
        "(cheaper, faster); FDR passes the full document to a large-context model "
        "(higher recall, higher cost). Iterating on the prompt or passage selector "
        "requires no Python rewrite. DataGatherer is a more specialized pipeline — "
        "strong defaults, less flexibility per change.\n"
    )

    parts.append(
        "## 10. Failure analysis\n"
        "See the failure-example sections above. Common categories:\n"
        "- **missed_identifier** — accession buried in a non-canonical section "
        "(e.g. References) or stated only in a supplementary file we did not "
        "download.\n"
        "- **hallucinated_identifier** — the LLM repeated an identifier from a "
        "tool/method citation rather than the paper's own data deposit.\n"
        "- **wrong_repository** — repository name absent in text and the prefix "
        "is ambiguous (DOIs in Zenodo vs Figshare vs Dryad).\n"
        "- **incomplete_identifier** — truncation when the accession spans a "
        "PDF line break.\n"
    )

    parts.append(
        "## 11. Reflection: RTR vs FDR vs specialized tools\n"
        "RTR trades recall for efficiency: it never misses a dataset that appears "
        "in a passage matched by the retrieval heuristic, but can miss ones buried "
        "in unexpected sections. FDR with a large-context model (Gemini 2.5 Flash, "
        "1M token window) sees the full text, improving recall at higher API cost. "
        "DataGatherer encodes more domain knowledge but is harder to retarget. "
        "The right choice depends on cost tolerance and the diversity of paper formats.\n"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path
