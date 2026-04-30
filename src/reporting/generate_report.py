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
    docetl_metrics_path: Path,
    datagatherer_metrics_path: Path | None,
    docetl_run_summary: dict[str, Any] | None,
    datagatherer_run_summary: dict[str, Any] | None,
    output_path: Path,
) -> Path:
    docetl_m = _maybe_load(docetl_metrics_path)
    dg_m = _maybe_load(datagatherer_metrics_path) if datagatherer_metrics_path else None

    parts: list[str] = []
    parts.append("# Dataset Reference Extraction — DocETL vs DataGatherer\n")
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

    parts.append("## 6. Metrics — DocETL\n")
    parts.append(_metrics_table(docetl_m, "DocETL"))
    parts.append(_failure_examples(docetl_m))

    parts.append("## 7. Metrics — DataGatherer (baseline)\n")
    if dg_m is None:
        parts.append("_DataGatherer baseline metrics not available._\n")
    else:
        parts.append(_metrics_table(dg_m, "DataGatherer"))
        parts.append(_failure_examples(dg_m))

    # Comparison
    parts.append("## 8. Comparison\n")
    if docetl_m and dg_m:
        d_p = docetl_m["pair_micro"]["f1"]
        g_p = dg_m["pair_micro"]["f1"]
        d_t = docetl_m["triple_micro"]["f1"]
        g_t = dg_m["triple_micro"]["f1"]
        parts.append(
            f"| Aspect | DocETL | DataGatherer |\n|---|---|---|\n"
            f"| Pair F1 (micro) | {d_p} | {g_p} |\n"
            f"| Triple F1 (micro) | {d_t} | {g_t} |\n"
            f"| Papers covered | {docetl_m['coverage']['n_papers_with_prediction']} | "
            f"{dg_m['coverage']['n_papers_with_prediction']} |\n"
        )
    else:
        parts.append("Side-by-side metrics will appear here once both runs are completed.\n")

    cost_lines = []
    if docetl_run_summary:
        cost_lines.append(f"- DocETL: {json.dumps(docetl_run_summary.get('cost', {}))}")
    if datagatherer_run_summary:
        cost_lines.append(f"- DataGatherer: {json.dumps({k: v for k, v in datagatherer_run_summary.items() if k != 'predictions_path'})}")
    if cost_lines:
        parts.append("\n**Cost / runtime:**\n" + "\n".join(cost_lines) + "\n")

    parts.append(
        "\n**Engineering effort:** the DocETL implementation is one YAML file "
        "(`pipelines/dataset_reference_extraction.yaml`) plus deterministic "
        "post-processing. Iterating on the prompt or the candidate-passage "
        "selector requires no Python rewrite. DataGatherer is a more "
        "specialized pipeline — strong defaults, less flexibility per change.\n"
    )

    parts.append(
        "## 9. Failure analysis\n"
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
        "## 10. Reflection: DocETL vs specialized tools\n"
        "DocETL trades a few percentage points of accuracy on edge cases for "
        "very fast iteration: the entire pipeline, including the LLM prompt, "
        "fits in one YAML. A specialized tool like DataGatherer encodes more "
        "domain knowledge but is harder to retarget. The right choice depends "
        "on whether the team owns the prompt loop or the curation rules.\n"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path
