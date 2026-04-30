from src.evaluation.match_records import match_pairs
from src.evaluation.metrics import categorize_failures, coverage, macro_metrics, micro_metrics


GT = [
    {"paper_id": "PMC1", "pmcid": "PMC1", "pmid": "", "paper_doi": "10.1/a",
     "dataset_identifier": "GSE100", "repository": "GEO"},
    {"paper_id": "PMC1", "pmcid": "PMC1", "pmid": "", "paper_doi": "10.1/a",
     "dataset_identifier": "PXD200", "repository": "PRIDE"},
    {"paper_id": "PMC2", "pmcid": "PMC2", "pmid": "", "paper_doi": "10.2/b",
     "dataset_identifier": "MSV300", "repository": "MassIVE"},
]
PRED = [
    # exact match
    {"paper_id": "PMC1", "pmcid": "PMC1", "paper_doi": "10.1/a",
     "dataset_identifier": "GSE100", "repository": "GEO"},
    # match by DOI alone (paper_id different)
    {"paper_id": "myPaper", "pmcid": "", "paper_doi": "10.1/a",
     "dataset_identifier": "PXD200", "repository": "PRIDE"},
    # hallucinated
    {"paper_id": "PMC1", "pmcid": "PMC1", "paper_doi": "10.1/a",
     "dataset_identifier": "GSE999", "repository": "GEO"},
    # missed: MSV300 in GT but not in predictions
]


def test_micro_pair_metrics():
    match = match_pairs(PRED, GT, repository_aware=False)
    m = micro_metrics(match)
    # 2 TPs, 1 FP, 1 FN
    assert m["tp"] == 2
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert 0 < m["precision"] < 1
    assert 0 < m["recall"] < 1


def test_macro_metrics_per_paper():
    match = match_pairs(PRED, GT, repository_aware=False)
    m = macro_metrics(match)
    assert m["n_papers"] == 2  # PMC1 and PMC2


def test_repository_aware_match_distinguishes_repo():
    pred_wrong_repo = [
        {"paper_id": "PMC1", "pmcid": "PMC1", "paper_doi": "10.1/a",
         "dataset_identifier": "GSE100", "repository": "ArrayExpress"},
    ]
    gt_one = [GT[0]]
    pair = match_pairs(pred_wrong_repo, gt_one, repository_aware=False)
    triple = match_pairs(pred_wrong_repo, gt_one, repository_aware=True)
    # pair-level: it's a match because (paper, dataset_id) align
    assert micro_metrics(pair)["tp"] == 1
    # triple-level: repository differs → no match
    assert micro_metrics(triple)["tp"] == 0


def test_coverage():
    cov = coverage(PRED, GT)
    assert cov["n_papers_with_prediction"] >= 1
    assert cov["n_papers_with_groundtruth"] == 2


def test_categorize_failures_buckets_fp_and_fn():
    match = match_pairs(PRED, GT, repository_aware=False)
    fails = categorize_failures(match)
    # MSV300 is missed
    assert any("msv300" in str(t).lower() for t in fails["missed_identifier"])
    # GSE999 is hallucinated
    assert any("gse999" in str(t).lower() for t in fails["hallucinated_identifier"])
