"""Run DataGatherer (https://github.com/VIDA-NYU/data-gatherer) as a baseline.

Verified against ``data_gatherer==0.2.1`` (April 2026).

Several gotchas in 0.2.1 that this wrapper papers over:

1. DataGatherer reads its OpenAI key from ``GPT_API_KEY`` (not the canonical
   ``OPENAI_API_KEY``). We alias both at module load so a project-standard
   ``.env`` works.
2. ``DataGatherer.process_articles(...)`` returns ``{preprocessed_url: pd.DataFrame, ...}``
   by default. We pass ``return_df_joint=True`` to receive a single combined
   DataFrame (with ``source_url`` as a column) — much easier to group on.
3. ``save_to_cache=True`` triggers a TypeError inside
   ``save_func_output_to_cache`` because the caller passes a list where the
   cache writer expects a dict. We always pass ``save_to_cache=False``.
4. The per-URL ``process_url`` swallows all exceptions and returns ``None``
   on failure. We detect "everything failed" by checking the joint DataFrame
   for emptiness and surface a clear error.
5. DataGatherer's URL-validation step uses a 0.5 s connect timeout that
   produces noisy WARNING lines for jPOST / PRIDE / ENA. We bump those
   loggers to ERROR.

Two strategies are exposed:

- ``strategy="retrieve"`` (default) → ``full_document_read=False``
  Retrieve-Then-Read: DataGatherer's own section retriever picks
  candidate passages, then prompts the LLM.
- ``strategy="full"`` → ``full_document_read=True``
  Full-Document Read: feed the entire paper to the LLM in one shot.
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from ..extraction.normalize_outputs import flatten_docetl_output, write_predictions

REPO_ROOT = Path(__file__).resolve().parents[2]


class DataGathererUnavailable(RuntimeError):
    pass


def _alias_api_keys() -> None:
    """DataGatherer reads GPT_API_KEY; mirror OPENAI_API_KEY into it.

    Also load .env at module level so importing this file from a script
    that hasn't already loaded dotenv still works.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except Exception:
        pass
    if not os.environ.get("GPT_API_KEY") and os.environ.get("OPENAI_API_KEY"):
        os.environ["GPT_API_KEY"] = os.environ["OPENAI_API_KEY"]


def _import_datagatherer():
    try:
        mod = importlib.import_module("data_gatherer.data_gatherer")
        return mod.DataGatherer
    except Exception as e:
        raise DataGathererUnavailable(
            "Could not import data-gatherer. Install with:\n"
            "  pip install git+https://github.com/VIDA-NYU/data-gatherer\n"
            f"Underlying error: {e}"
        )


def _pick_url(paper: dict[str, Any]) -> str | None:
    """Pick the most informative URL we can use for one of our papers."""
    if paper.get("source_url"):
        return paper["source_url"]
    pmcid = paper.get("pmcid")
    if pmcid:
        return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
    doi = paper.get("paper_doi")
    if doi:
        return f"https://doi.org/{doi}"
    return None


def _coerce_one_ref(item: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one DataGatherer DataFrame row into our internal ref schema."""
    ds_id = (
        item.get("dataset_identifier")
        or item.get("dataset_id")
        or item.get("identifier")
        or item.get("id")
        or ""
    )
    if not ds_id or str(ds_id).lower() in ("nan", "none", "n/a"):
        return None
    repo = (
        item.get("data_repository")
        or item.get("repository")
        or item.get("source")
        or ""
    )
    page = item.get("dataset_webpage") or ""
    return {
        "dataset_identifier": str(ds_id),
        "repository": str(repo) if repo and str(repo).lower() not in ("nan", "none") else "",
        "evidence": "",
        "confidence": "",
        "notes": f"datagatherer; webpage={page}" if page and page != "n/a" else "",
    }


def _df_to_records(df: Any, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group a DataGatherer joint DataFrame by paper URL.

    The DataFrame returned with ``return_df_joint=True`` carries one row per
    extracted dataset reference, plus a ``source_url`` column we can group
    on. Anything else (e.g. ``pub_title``) is ignored.
    """
    try:
        import pandas as pd
    except Exception:  # pragma: no cover
        pd = None  # type: ignore[assignment]

    by_url: dict[str, list[dict[str, Any]]] = {}
    if pd is not None and isinstance(df, pd.DataFrame) and len(df) > 0:
        url_col = next(
            (c for c in ("source_url", "paper_url", "publication_url", "url")
             if c in df.columns),
            None,
        )
        if url_col is None:
            url_col = next((c for c in df.columns if "url" in c.lower()), None)
        for _, row in df.iterrows():
            url = (row.get(url_col) if url_col else "") or ""
            ref = _coerce_one_ref(row.to_dict())
            if ref is not None:
                by_url.setdefault(str(url), []).append(ref)

    out: list[dict[str, Any]] = []
    for paper in papers:
        url = _pick_url(paper) or ""
        refs = by_url.get(url, [])
        out.append({**paper, "dataset_references": refs})
    return out


def run_datagatherer(
    input_path: Path,
    output_path: Path,
    strategy: str = "retrieve",
    llm_name: str | None = None,
) -> dict[str, Any]:
    """Run DataGatherer over our DocETL-style paper records.

    Args:
        input_path: papers.json from scripts/prepare_inputs.py or fetch_urls.py.
        output_path: predictions JSONL path; same schema as DocETL's output.
        strategy: ``"retrieve"`` (Retrieve-Then-Read) or ``"full"`` (Full-Document Read).
        llm_name: LiteLLM-compatible model. Defaults to DOCETL_MODEL or ``gpt-4o-mini``.
    """
    _alias_api_keys()

    input_path = Path(input_path)
    output_path = Path(output_path)
    papers = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(papers, list):
        raise ValueError("Expected a JSON array of papers")

    status: dict[str, Any] = {
        "tool": "datagatherer",
        "strategy": strategy,
        "n_papers": len(papers),
    }

    if not os.environ.get("GPT_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        status["status"] = "no_api_key"
        status["error"] = (
            "Neither OPENAI_API_KEY nor GPT_API_KEY is set in the environment. "
            "Set OPENAI_API_KEY in .env (this wrapper aliases it to GPT_API_KEY)."
        )
        write_predictions([], output_path)
        return status

    try:
        DataGatherer = _import_datagatherer()
    except DataGathererUnavailable as e:
        status["status"] = "unavailable"
        status["error"] = str(e)
        write_predictions([], output_path)
        return status

    full_document_read = strategy == "full"
    model = llm_name or os.environ.get("DOCETL_MODEL") or "gpt-4o-mini"

    # Build URL list, drop papers we can't address.
    url_list: list[str] = []
    paper_for_url: list[dict[str, Any]] = []
    for paper in papers:
        url = _pick_url(paper)
        if url:
            url_list.append(url)
            paper_for_url.append(paper)
    if not url_list:
        status["status"] = "no_urls"
        status["error"] = "No paper has a usable URL (source_url / pmcid / doi)."
        write_predictions([], output_path)
        return status

    # Quiet noisy URL-validation warnings (0.5 s connect timeout to jPOST etc.)
    for noisy in ("base_parser", "data_fetcher"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    # Keep the data_gatherer logger at ERROR so per-URL failures still surface.
    logging.getLogger("data_gatherer").setLevel(logging.ERROR)

    print(f"DataGatherer: model={model}  strategy={strategy}  n_urls={len(url_list)}")
    start = time.time()
    try:
        dg = DataGatherer(
            llm_name=model,
            process_entire_document=full_document_read,
            log_level=40,
            save_to_cache=False,   # workaround for the 0.2.1 cache TypeError bug
            load_from_cache=False,
        )
        # return_df_joint=True gives us a single combined DataFrame instead of
        # the default {url: per-url DataFrame, ...} dict. Easier to handle.
        df = dg.process_articles(
            url_list=url_list,
            full_document_read=full_document_read,
            headless=True,
            use_portkey=False,
            return_df_joint=True,
        )
    except ValueError as e:
        # process_articles raises "All objects passed were None" when *every*
        # URL crashed inside process_url. That's the canonical "complete
        # failure" signal we want to surface.
        status["status"] = "all_urls_failed"
        status["error"] = (
            f"DataGatherer failed on every URL. Underlying message: {e}. "
            "Check stderr above for the per-URL traceback (commonly: invalid "
            "API key, model name, or network access to NCBI E-utilities)."
        )
        status["elapsed_sec"] = round(time.time() - start, 1)
        write_predictions([], output_path)
        return status
    except Exception as e:
        status["status"] = "error"
        status["error"] = f"{type(e).__name__}: {e}"
        status["elapsed_sec"] = round(time.time() - start, 1)
        write_predictions([], output_path)
        return status

    # Sanity check: empty DataFrame means DataGatherer ran but extracted
    # nothing — usually a model/parser issue, not a code issue. Let the user
    # see the row count so they don't guess.
    raw_rows = int(len(df)) if df is not None and hasattr(df, "__len__") else 0
    grouped = _df_to_records(df, paper_for_url)
    flat = flatten_docetl_output(grouped)
    write_predictions(flat, output_path)

    status.update({
        "status": "ok" if flat else "no_predictions",
        "elapsed_sec": round(time.time() - start, 1),
        "n_predictions": len(flat),
        "predictions_path": str(output_path),
        "raw_df_rows": raw_rows,
    })
    if not flat:
        status["error"] = (
            "DataGatherer returned a DataFrame with no usable rows. Common "
            "causes: model error suppressed inside process_url (rerun with "
            "log_level=20 to see), or every paper genuinely had no datasets."
        )
    return status
