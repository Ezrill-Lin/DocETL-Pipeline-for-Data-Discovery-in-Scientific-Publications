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


def test_flatten_emits_na_for_empty_or_missing_identifiers():
    """Blank / missing identifiers become a single N/A row per paper."""
    rows = [
        {
            "paper_id": "PMC2",
            "paper_doi": "",
            "dataset_references": [
                {"dataset_identifier": "", "repository": "GEO",
                 "evidence": "data deposited in GEO", "notes": "no accession"},
                {"dataset_identifier": "PXD000001", "repository": "PRIDE",
                 "evidence": "PXD000001 in PRIDE"},
            ],
        },
        {
            "paper_id": "PMC3",
            "dataset_references": [],
        },
    ]
    out = flatten_docetl_output(rows)
    assert any(r["dataset_identifier"] == "PXD000001" and r["repository"] == "PRIDE" for r in out)
    # The blank-id row from PMC2 became an N/A row carrying the named repo.
    assert any(
        r["paper_id"] == "PMC2" and r["dataset_identifier"] == "N/A" and r["repository"] == "GEO"
        for r in out
    )
    # PMC3 (no refs) gets a single N/A placeholder.
    pmc3_rows = [r for r in out if r["paper_id"] == "PMC3"]
    assert len(pmc3_rows) == 1
    assert pmc3_rows[0]["dataset_identifier"] == "N/A"


def test_flatten_drops_schema_violations_and_inconsistent_evidence():
    """Bare prefixes and identifiers contradicted by quoted evidence are dropped."""
    rows = [
        {
            "paper_id": "PMC4",
            "dataset_references": [
                # Bare prefix — fails GEO schema regex.
                {"dataset_identifier": "GSE", "repository": "GEO",
                 "evidence": "data deposited in GEO repository"},
                # Evidence quotes a *different* accession than the one
                # emitted — strong hallucination signal, drop.
                {"dataset_identifier": "GSE99999", "repository": "GEO",
                 "evidence": "data are at GSE12345 in GEO"},
                # Real, well-formed, evidence quotes the same accession.
                {"dataset_identifier": "GSE54321", "repository": "GEO",
                 "evidence": "deposited GSE54321 in GEO"},
                # Descriptive prose with no accession tokens — accept,
                # since we can't verify and don't want to punish under-quoting.
                {"dataset_identifier": "PXD000111", "repository": "PRIDE",
                 "evidence": "deposited to PRIDE Archive via ProteomeXchange"},
            ],
        }
    ]
    out = flatten_docetl_output(rows)
    pmc4 = [r for r in out if r["paper_id"] == "PMC4"]
    ids = sorted(r["dataset_identifier"] for r in pmc4)
    assert ids == ["GSE54321", "PXD000111"], ids


def test_flatten_dedups_proteomexchange_doi_alias():
    """`10.6019/PXD#####` is the same dataset as the bare `PXD#####`."""
    rows = [
        {
            "paper_id": "PMC5",
            "dataset_references": [
                {"dataset_identifier": "PXD029805", "repository": "PRIDE",
                 "evidence": "deposited PXD029805 to PRIDE"},
                {"dataset_identifier": "10.6019/PXD029805", "repository": "ProteomeXchange",
                 "evidence": "10.6019/PXD029805 in ProteomeCentral"},
            ],
        }
    ]
    out = flatten_docetl_output(rows)
    pmc5 = [r for r in out if r["paper_id"] == "PMC5"]
    assert len(pmc5) == 1
    assert pmc5[0]["dataset_identifier"] == "PXD029805"


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


def test_normalize_groundtruth_row_preserves_na():
    """GT rows where identifier is blank / N/A / NA collapse to sentinel N/A."""
    for raw in ["", "N/A", "n/a", "NA", "none"]:
        row = {"PMCID": "PMC9", "accession": raw, "database": "AstraZeneca"}
        out = normalize_groundtruth_row(row)
        assert out["dataset_identifier"] == "N/A", raw
        # Free-form repository is preserved verbatim (after strip) so it
        # can match against a prediction of the same name.
        assert out["repository"] == "AstraZeneca"
