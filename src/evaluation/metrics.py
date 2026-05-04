"""Compute precision / recall / F1 from match results."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..extraction.registry import is_na_identifier


def prf1(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def micro_metrics(match_result: dict[str, Any]) -> dict[str, float]:
    tp = len(match_result["true_positives"])
    fp = len(match_result["false_positives"])
    fn = len(match_result["false_negatives"])
    metrics = prf1(tp, fp, fn)
    metrics.update({"tp": tp, "fp": fp, "fn": fn})
    return metrics


def macro_metrics(match_result: dict[str, Any]) -> dict[str, float]:
    """Per-paper macro-average. Uses paper-key (first tuple element) as the group."""
    by_paper_tp: dict[str, int] = defaultdict(int)
    by_paper_fp: dict[str, int] = defaultdict(int)
    by_paper_fn: dict[str, int] = defaultdict(int)
    for t in match_result["true_positives"]:
        by_paper_tp[t[0]] += 1
    for t in match_result["false_positives"]:
        by_paper_fp[t[0]] += 1
    for t in match_result["false_negatives"]:
        by_paper_fn[t[0]] += 1
    papers = set(by_paper_tp) | set(by_paper_fp) | set(by_paper_fn)
    if not papers:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_papers": 0}
    total_p = total_r = total_f = 0.0
    for paper in papers:
        m = prf1(by_paper_tp[paper], by_paper_fp[paper], by_paper_fn[paper])
        total_p += m["precision"]
        total_r += m["recall"]
        total_f += m["f1"]
    n = len(papers)
    return {
        "precision": round(total_p / n, 4),
        "recall": round(total_r / n, 4),
        "f1": round(total_f / n, 4),
        "n_papers": n,
    }


def coverage(predictions: list[dict[str, Any]], groundtruth: list[dict[str, Any]]) -> dict[str, int]:
    real_preds = [r for r in predictions if not is_na_identifier(r.get("dataset_identifier"))]
    na_preds = [r for r in predictions if is_na_identifier(r.get("dataset_identifier")) and r.get("paper_id")]
    pred_papers = {r.get("paper_id", "") for r in real_preds}
    gt_papers = {r.get("paper_id", "") for r in groundtruth if r.get("dataset_identifier")}
    empty_outputs = sum(
        1 for r in predictions if not r.get("dataset_identifier")
    )
    return {
        "n_papers_with_prediction": len(pred_papers),
        "n_papers_with_groundtruth": len(gt_papers),
        "n_empty_outputs": empty_outputs,
        "n_total_predictions": len(predictions),
        "n_real_predictions": len(real_preds),
        "n_na_predictions": len(na_preds),
        "n_total_groundtruth": len(groundtruth),
    }


def categorize_failures(match_result: dict[str, Any]) -> dict[str, list]:
    """Bucket FPs and FNs into rough categories.

    Categories:
    - hallucinated_identifier: FP with a well-formed accession that isn't in GT
    - wrong_repository:        FP whose (paper, dataset) IS in GT but repo differs
                               (only meaningful when repository_aware match was used)
    - missed_identifier:       FN
    - incomplete_identifier:   FP whose dataset_id is shorter than typical accession
    """
    out: dict[str, list] = {
        "missed_identifier": list(match_result["false_negatives"]),
        "hallucinated_identifier": [],
        "incomplete_identifier": [],
        "wrong_repository": [],
    }
    fp = match_result["false_positives"]
    fn_keys_2 = {(t[0], t[1]) for t in match_result["false_negatives"]}
    for t in fp:
        # repository-aware: tuple is (paper, ds, repo); FP that matches GT without repo means wrong_repository
        if len(t) == 3 and (t[0], t[1]) not in fn_keys_2:
            # the (paper, ds) does NOT appear in FN list; it's not present in GT under any repo → hallucinated
            out["hallucinated_identifier"].append(t)
        elif len(t) == 3:
            out["wrong_repository"].append(t)
        else:
            ds = t[1]
            # Heuristic: very short ids look incomplete (e.g. "GSE")
            if ds and len(ds) <= 4 and ds.isalpha():
                out["incomplete_identifier"].append(t)
            else:
                out["hallucinated_identifier"].append(t)
    return out
