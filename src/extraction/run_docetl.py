"""Run the DocETL pipeline programmatically.

We render placeholders in the pipeline YAML, write a temporary file, and hand
it to docetl.runner.DSLRunner. The runner writes its raw output to the path
specified by `pipeline.output.path`. We then load that output and flatten it
through src/extraction/normalize_outputs.py.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .normalize_outputs import flatten_docetl_output, write_predictions
from .registry import prompt_repository_block

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE = REPO_ROOT / "pipelines" / "pipeline_rtr.yaml"


def _render(text: str, mapping: dict[str, str]) -> str:
    for key, value in mapping.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def _approx_token_cost(papers: list[dict[str, Any]], settings: dict[str, Any]) -> dict[str, Any]:
    chars = sum(len(p.get("candidate_passages", "")) for p in papers)
    tokens_per_char = settings.get("approx_input_tokens_per_char", 0.25)
    in_price = settings.get("default_input_price_per_1k", 0.00015)
    out_price = settings.get("default_output_price_per_1k", 0.0006)
    in_tokens = int(chars * tokens_per_char)
    # Crude estimate: outputs are ~10% the size of inputs
    out_tokens = int(in_tokens * 0.1)
    return {
        "input_chars": chars,
        "approx_input_tokens": in_tokens,
        "approx_output_tokens": out_tokens,
        "approx_cost_usd": round(
            (in_tokens / 1000) * in_price + (out_tokens / 1000) * out_price, 6
        ),
    }


def run_pipeline(
    input_path: Path,
    output_path: Path,
    pipeline_yaml: Path = DEFAULT_PIPELINE,
    model: str | None = None,
    intermediate_dir: Path | None = None,
    cost_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render the YAML, run DocETL, and emit normalized predictions.

    Returns a dict with paths and cost stats.
    """
    load_dotenv()

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    pipeline_yaml = Path(pipeline_yaml).resolve()
    model = model or os.environ.get("DOCETL_MODEL") or "gpt-4o-mini"
    intermediate_dir = Path(
        intermediate_dir or REPO_ROOT / "data" / "processed" / ".docetl_cache"
    ).resolve()
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output = output_path.with_suffix(".raw.json")

    rendered = _render(
        pipeline_yaml.read_text(encoding="utf-8"),
        {
            # Use absolute paths to resolve the path difference between Linux/Mac and Windows.
            "INPUT_PATH": input_path.as_posix(),
            "OUTPUT_PATH": raw_output.as_posix(),
            "INTERMEDIATE_DIR": intermediate_dir.as_posix(),
            "MODEL": model,
            # The registry is the single source of truth for repositories.
            # Render the per-repo bullet list into the prompt at run time so
            # adding a repository to repositories.yaml propagates automatically.
            # Indent matches the surrounding `prompt: |` block scalar.
            "REPOSITORIES": prompt_repository_block(indent="      "),
        },
    )

    # Sanity: parse it before sending to DocETL so we get clear errors.
    yaml.safe_load(rendered)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(rendered)
        tmp_path = Path(tf.name)

    # Lazy import: docetl pulls in heavy deps.
    from docetl.runner import DSLRunner

    total_cost = 0.0
    try:
        runner = DSLRunner.from_yaml(str(tmp_path))
        try:
            total_cost = runner.load_run_save()
        except Exception as e:
            raise RuntimeError(f"DocETL pipeline execution failed: {e}") from e
    finally:
        tmp_path.unlink(missing_ok=True)

    # Read raw outputs (DocETL writes a JSON array to `path`)
    if not raw_output.exists():
        raise FileNotFoundError(f"DocETL did not produce expected output: {raw_output}")
    raw = json.loads(raw_output.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Unexpected DocETL output format (expected a list).")

    # Approximate cost from inputs (kept as a fallback when DocETL doesn't
    # report token usage in its writeback).
    papers = json.loads(input_path.read_text(encoding="utf-8"))
    cost_estimate = _approx_token_cost(papers if isinstance(papers, list) else [], cost_settings or {})
    cost_estimate["docetl_reported_cost_usd"] = round(float(total_cost or 0.0), 6)

    rows = flatten_docetl_output(raw)
    write_predictions(rows, output_path)

    return {
        "raw_output_path": str(raw_output),
        "predictions_path": str(output_path),
        "n_papers": len(papers if isinstance(papers, list) else []),
        "n_predictions": len(rows),
        "cost": cost_estimate,
    }
