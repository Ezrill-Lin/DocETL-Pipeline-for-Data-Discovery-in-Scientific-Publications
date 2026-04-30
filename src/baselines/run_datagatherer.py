"""Run DataGatherer (https://github.com/VIDA-NYU/data-gatherer) as a baseline.

DataGatherer's public API has shifted over recent releases. We try the most
common entry points by name; if none work we surface a clear error and a
manual-fallback hint. The output is normalized to the same prediction schema
emitted by the DocETL pipeline so the same evaluation code can score both.

Two strategies are supported when the package exposes them:
- "full"     — Full-Document Read
- "retrieve" — Retrieve-Then-Read
"""
from __future__ import annotations

import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

from ..extraction.normalize_outputs import flatten_docetl_output, write_predictions

REPO_ROOT = Path(__file__).resolve().parents[2]


class DataGathererUnavailable(RuntimeError):
    pass


def _import_datagatherer():
    """Try a handful of plausible module paths."""
    for modname in ("data_gatherer", "datagatherer"):
        try:
            return importlib.import_module(modname)
        except Exception:
            continue
    raise DataGathererUnavailable(
        "Could not import data-gatherer. Install with:\n"
        "  pip install git+https://github.com/VIDA-NYU/data-gatherer\n"
        "or clone the repo and `pip install -e .` from its root."
    )


def _build_extractor(strategy: str):
    """Return a callable extractor(text|path, paper_id) -> list[dict].

    We probe the package surface for known class/function names. If the API
    has changed, edit this function — the rest of the wrapper is generic.
    """
    pkg = _import_datagatherer()

    # Heuristic 1: Pipeline-style entry point
    for cls_name in ("DataGatherer", "Pipeline", "DataGathererPipeline"):
        cls = getattr(pkg, cls_name, None)
        if cls is not None:
            try:
                instance = cls(strategy=strategy)  # type: ignore[call-arg]
            except TypeError:
                instance = cls()
            for method in ("extract", "run", "predict", "process_paper"):
                fn = getattr(instance, method, None)
                if callable(fn):
                    return lambda text, paper_id, fn=fn: fn(text, paper_id=paper_id)

    # Heuristic 2: top-level function
    for fn_name in ("extract_dataset_references", "extract", "run"):
        fn = getattr(pkg, fn_name, None)
        if callable(fn):
            return lambda text, paper_id, fn=fn: fn(text, paper_id=paper_id)

    raise DataGathererUnavailable(
        "data-gatherer is installed but no recognized entry point was found. "
        "Edit src/baselines/run_datagatherer.py:_build_extractor to point at "
        "the correct class/function for your version, or run DataGatherer "
        "manually and place its predictions at "
        "data/predictions/datagatherer_predictions.jsonl."
    )


def _coerce_refs(raw: Any) -> list[dict[str, Any]]:
    """Normalize whatever DataGatherer returns into our list-of-dicts schema."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        # If it looks like {"dataset_references": [...]}, unwrap.
        for key in ("dataset_references", "datasets", "references", "results"):
            if key in raw and isinstance(raw[key], list):
                raw = raw[key]
                break
        else:
            return []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        out.append({
            "dataset_identifier": (
                item.get("dataset_identifier")
                or item.get("accession")
                or item.get("identifier")
                or item.get("id")
                or ""
            ),
            "repository": (
                item.get("repository")
                or item.get("database")
                or item.get("source")
                or ""
            ),
            "evidence": item.get("evidence") or item.get("context") or "",
            "confidence": str(item.get("confidence") or ""),
            "notes": item.get("notes") or "",
        })
    return out


def run_datagatherer(
    input_path: Path,
    output_path: Path,
    strategy: str = "full",
) -> dict[str, Any]:
    """Run DataGatherer over a DocETL-style JSON input file.

    Falls back gracefully — if DataGatherer isn't usable on this machine, we
    write an empty predictions file and a status note rather than crashing.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    papers = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(papers, list):
        raise ValueError("Expected a JSON array of papers")

    status = {"status": "ok", "strategy": strategy, "n_papers": len(papers)}
    rows: list[dict[str, Any]] = []
    start = time.time()

    try:
        extractor = _build_extractor(strategy)
    except DataGathererUnavailable as e:
        status["status"] = "unavailable"
        status["error"] = str(e)
        # Write empty predictions so evaluation still runs.
        write_predictions([], output_path)
        return status

    for paper in papers:
        text = paper.get("full_text") or paper.get("candidate_passages") or ""
        try:
            raw = extractor(text, paper.get("paper_id", ""))
        except Exception as e:
            print(f"[WARN] DataGatherer failed on {paper.get('paper_id', '?')}: {e}")
            raw = []
        refs = _coerce_refs(raw)
        rows.append({**paper, "dataset_references": refs})

    flat = flatten_docetl_output(rows)
    write_predictions(flat, output_path)

    status.update({
        "elapsed_sec": round(time.time() - start, 2),
        "n_predictions": len(flat),
        "predictions_path": str(output_path),
    })
    return status
