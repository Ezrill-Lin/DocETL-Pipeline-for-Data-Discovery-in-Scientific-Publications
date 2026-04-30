from src.extraction.url_builder import (
    build_dataset_url,
    infer_repository,
    normalize_identifier,
    normalize_repository,
)


def test_normalize_identifier_strips_punctuation():
    assert normalize_identifier(" GSE12345. ") == "GSE12345"
    assert normalize_identifier("'PXD009876',") == "PXD009876"


def test_normalize_identifier_strips_doi_prefix():
    assert normalize_identifier("https://doi.org/10.5061/dryad.abc123") == "10.5061/dryad.abc123"
    assert normalize_identifier("doi:10.5281/zenodo.1") == "10.5281/zenodo.1"


def test_normalize_identifier_uppercases_known_prefixes():
    assert normalize_identifier("gse12345") == "GSE12345"
    assert normalize_identifier("pxd000001") == "PXD000001"
    assert normalize_identifier("e-mtab-1234") == "E-MTAB-1234"


def test_normalize_repository_aliases():
    assert normalize_repository("Gene Expression Omnibus") == "GEO"
    assert normalize_repository("PRIDE Archive") == "PRIDE"
    assert normalize_repository("Massive") == "MassIVE"
    assert normalize_repository("Panorama Public") == "PanoramaPublic"
    assert normalize_repository("ProteomeCentral") == "ProteomeXchange"
    assert normalize_repository("") == ""


def test_infer_repository_from_prefix():
    assert infer_repository("GSE12345", None) == "GEO"
    # PXD-prefixed IDs are PRIDE accession numbers (deposited via ProteomeXchange)
    assert infer_repository("PXD009876", "") == "PRIDE"
    assert infer_repository("MSV000001", None) == "MassIVE"
    assert infer_repository("PRJNA12345", None) == "BioProject"


def test_infer_repository_uses_explicit_when_known():
    # Explicit repo wins when it resolves to a known canonical name
    assert infer_repository("PXD009876", "ProteomeXchange") == "ProteomeXchange"


def test_build_dataset_url_geo():
    url = build_dataset_url("gse12345", "GEO")
    assert url == "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE12345"


def test_build_dataset_url_pride():
    url = build_dataset_url("PXD009876", "PRIDE")
    assert url == "https://www.ebi.ac.uk/pride/archive/projects/PXD009876"


def test_build_dataset_url_proteomexchange_explicit():
    url = build_dataset_url("PXD009876", "ProteomeXchange")
    assert url == "https://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD009876"


def test_build_dataset_url_doi_zenodo():
    url = build_dataset_url("10.5281/zenodo.1234567", "Zenodo")
    assert url == "https://doi.org/10.5281/zenodo.1234567"


def test_build_dataset_url_arrayexpress():
    url = build_dataset_url("E-MTAB-1234", "ArrayExpress")
    assert url == "https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-1234"


def test_build_dataset_url_unknown_returns_empty():
    # Non-DOI string with no recognized prefix and no repo
    assert build_dataset_url("foobar123", None) == ""


def test_url_not_invented_for_unknown_repo():
    # If repository is unknown but identifier looks like a DOI, we still build via DOI
    assert build_dataset_url("10.1234/x", "SomeUnknownRepo") == "https://doi.org/10.1234/x"
    # Non-DOI accession with unknown repo: empty
    assert build_dataset_url("XYZ123", "SomeUnknownRepo") == ""
