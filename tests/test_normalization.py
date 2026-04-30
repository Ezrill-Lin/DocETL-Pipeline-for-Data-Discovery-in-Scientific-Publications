from src.extraction.normalize_outputs import flatten_docetl_output, normalize_record
from src.evaluation.load_groundtruth import normalize_groundtruth_row


def test_normalize_record_builds_url_and_canonicalizes_repo():
    paper = {"paper_id": "PMC1", "paper_doi": "10.1/abc", "pmcid": "PMC1", "pmid": ""}
    item = {
        "dataset_identifier": " gse12345 ",
        "repository": "Gene Expression Omnibus",
        "evidence": "deposited in GEO under accession GSE12345",
        "confidence": "high",
        "notes": "",
    }
    out = normalize_record(paper, item)
    assert out["dataset_identifier"] == "GSE12345"
    assert out["repository"] == "GEO"
    assert out["url"].endswith("acc=GSE12345")
    assert out["paper_id"] == "PMC1"
    assert out["confidence"] == "high"


def test_flatten_skips_empty_identifiers():
    rows = [
        {
            "paper_id": "PMC2",
            "paper_doi": "",
            "dataset_references": [
                {"dataset_identifier": "", "repository": "GEO"},
                {"dataset_identifier": "PXD000001", "repository": "PRIDE"},
            ],
        },
        {
            "paper_id": "PMC3",
            "dataset_references": [],
        },
    ]
    out = flatten_docetl_output(rows)
    # PMC2 → 1 valid row; PMC3 → placeholder empty row for coverage
    assert any(r["dataset_identifier"] == "PXD000001" and r["repository"] == "PRIDE" for r in out)
    assert any(r["paper_id"] == "PMC3" and r["dataset_identifier"] == "" for r in out)


def test_normalize_groundtruth_row_handles_aliases():
    row = {
        "PMCID": "1234567",
        "DOI": "https://doi.org/10.1/x",
        "accession": "gse9999",
        "database": "Gene Expression Omnibus",
    }
    out = normalize_groundtruth_row(row)
    assert out["pmcid"] == "PMC1234567"
    assert out["paper_doi"] == "10.1/x"
    assert out["dataset_identifier"] == "GSE9999"
    assert out["repository"] == "GEO"
