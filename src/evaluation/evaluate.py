"""Top-level evaluation entry point."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .load_groundtruth import load_groundtruth, normalize_groundtruth_row
from .match_records import match_pairs
from .metrics import categorize_failures, coverage, macro_metrics, micro_metrics


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("rows", [])
    else:
        raise ValueError(f"Unsupported predictions format: {p}")
    # Normalize so matching uses the same keys as groundtruth.
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(normalize_groundtruth_row(r))
    return out


def evaluate(
    predictions_path: Path,
    groundtruth_path: Path,
    output_path: Path | None = None,
    label: str = "docetl",
    model: str = "",
    benchmark: str = "",
) -> dict[str, Any]:
    predictions = _load_predictions(predictions_path)
    groundtruth = load_groundtruth(groundtruth_path)

    pair_match = match_pairs(predictions, groundtruth, repository_aware=False)
    triple_match = match_pairs(predictions, groundtruth, repository_aware=True)

    summary = {
        "label": label,
        "model": model,
        "benchmark": benchmark,
        "predictions_path": str(predictions_path),
        "groundtruth_path": str(groundtruth_path),
        "pair_micro": micro_metrics(pair_match),
        "pair_macro": macro_metrics(pair_match),
        "triple_micro": micro_metrics(triple_match),
        "triple_macro": macro_metrics(triple_match),
        "coverage": coverage(predictions, groundtruth),
        "failures": {
            "pair": categorize_failures(pair_match),
            "triple": categorize_failures(triple_match),
        },
    }
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
