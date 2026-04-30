"""Match predicted dataset references to ground-truth references."""
from __future__ import annotations

from typing import Any

from ..extraction.url_builder import normalize_identifier, normalize_repository
from .load_groundtruth import paper_key


def build_paper_index(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Build alias → canonical-paper-key index from a set of rows.

    Lets a predicted PMCID match a ground-truth row keyed by DOI when both
    were observed in the *same* dataset (predictions or groundtruth).
    """
    index: dict[str, str] = {}
    # Group rows by canonical key first
    canonical: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        canonical.setdefault(paper_key(r), []).append(r)
    for ckey, items in canonical.items():
        for r in items:
            for field in ("pmcid", "pmid", "paper_doi", "paper_id"):
                v = (r.get(field) or "").strip().lower()
                if v:
                    index[f"{field}:{v}"] = ckey
    return index


def resolve_paper(row: dict[str, Any], indexes: list[dict[str, str]]) -> str:
    """Try every alias of a paper across multiple indexes to find a shared key."""
    for field in ("pmcid", "pmid", "paper_doi", "paper_id"):
        v = (row.get(field) or "").strip().lower()
        if not v:
            continue
        alias = f"{field}:{v}"
        for idx in indexes:
            if alias in idx:
                return idx[alias]
    return paper_key(row)


def match_pairs(
    predictions: list[dict[str, Any]],
    groundtruth: list[dict[str, Any]],
    repository_aware: bool = False,
) -> dict[str, Any]:
    """Compute matched / missed / spurious sets at the (paper, dataset) level.

    Returns dicts of true_positives, false_positives, false_negatives. Each
    item is a tuple (canonical_paper_key, dataset_id[, repo]) so callers can
    enumerate failures for the report.
    """
    pred_index = build_paper_index(predictions)
    gt_index = build_paper_index(groundtruth)

    def _key(row: dict[str, Any]) -> tuple[str, ...]:
        ckey = resolve_paper(row, [gt_index, pred_index])
        ds = normalize_identifier(row.get("dataset_identifier", "")).lower()
        repo = normalize_repository(row.get("repository", ""))
        if repository_aware:
            return (ckey, ds, repo)
        return (ckey, ds)

    pred_set: set[tuple[str, ...]] = set()
    for r in predictions:
        ds = normalize_identifier(r.get("dataset_identifier", ""))
        if not ds:
            continue
        pred_set.add(_key(r))
    gt_set: set[tuple[str, ...]] = set()
    for r in groundtruth:
        ds = normalize_identifier(r.get("dataset_identifier", ""))
        if not ds:
            continue
        gt_set.add(_key(r))

    tp = pred_set & gt_set
    fp = pred_set - gt_set
    fn = gt_set - pred_set
    return {
        "true_positives": sorted(tp),
        "false_positives": sorted(fp),
        "false_negatives": sorted(fn),
        "n_predictions": len(pred_set),
        "n_groundtruth": len(gt_set),
    }
