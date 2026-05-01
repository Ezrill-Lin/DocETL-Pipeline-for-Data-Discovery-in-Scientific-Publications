from src.extraction.registry import (
    accession_pattern,
    is_na_identifier,
    prompt_repository_block,
    repository_pattern,
    synonym_group,
    validate_identifier,
)


def test_accession_pattern_covers_well_known_repos():
    pat = accession_pattern()
    for s in [
        "GSE12345", "PXD009876", "MSV000001", "phs001249.v1.p1",
        "EGAS00001000299", "PRJNA306801", "SAMN00012345", "IPX0004230000",
        "PDC000234",
    ]:
        assert pat.search(s), f"pattern should match {s}"


def test_repository_pattern_covers_aliases():
    pat = repository_pattern()
    for s in [
        "Gene Expression Omnibus", "GEO", "dbGaP", "EGA",
        "European Genome Phenome Archive", "Mendeley Data",
        "Proteomic Data Commons", "PRIDE",
    ]:
        assert pat.search(s), f"repo pattern should match {s}"


def test_synonym_group_treats_pride_and_proteomexchange_as_equivalent():
    assert synonym_group("PRIDE") == synonym_group("ProteomeXchange")
    assert synonym_group("Gene Expression Omnibus") == synonym_group("GEO")
    # Free-form labels fall back to lowercased form so they still compare.
    assert synonym_group("AstraZeneca") == "astrazeneca"
    assert synonym_group("") == ""
    assert synonym_group(None) == ""


def test_validate_identifier_rejects_bare_prefixes_and_shape_mismatches():
    assert validate_identifier("GSE12345", "GEO")
    assert not validate_identifier("GSE", "GEO")
    assert validate_identifier("PXD009876", "PRIDE")
    assert not validate_identifier("PXD", "PRIDE")
    assert validate_identifier("phs001249.v1.p1", "dbGaP")
    assert not validate_identifier("phs", "dbGaP")
    # No schema declared → accept (we don't fail closed on unknown shapes).
    assert validate_identifier("anything-goes", "PanoramaPublic")


def test_is_na_identifier_recognizes_common_blanks():
    for s in ["", "N/A", "n/a", "NA", "none", "null", None, "  "]:
        assert is_na_identifier(s), f"{s!r} should be N/A-equivalent"
    for s in ["GSE1", "PXD1", "X"]:
        assert not is_na_identifier(s)


def test_prompt_repository_block_renders_indented():
    block = prompt_repository_block(indent="      ")
    # Multi-line, every continuation line carries the indent.
    lines = block.split("\n")
    assert lines[0].startswith("  - ")
    for ln in lines[1:]:
        assert ln.startswith("        - "), ln
