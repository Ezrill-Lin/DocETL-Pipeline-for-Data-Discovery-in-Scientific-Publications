"""Parse PDF scientific papers into structured records."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .parse_html import DOI_RE, PMCID_RE, PMID_RE


def _import_fitz():
    try:
        import fitz  # PyMuPDF
        return fitz
    except ImportError:
        return None


def _import_pdfplumber():
    try:
        import pdfplumber
        return pdfplumber
    except ImportError:
        return None


HEADING_PATTERNS = [
    re.compile(r"^\s*\d+(?:\.\d+)*\s+[A-Z][A-Za-z0-9 ,\-/]{2,80}\s*$"),
    re.compile(r"^\s*(Abstract|Introduction|Methods?|Materials and Methods|Results?|"
               r"Discussion|Conclusions?|References|Acknowledg(?:e?ments?)|"
               r"Data Availability(?: Statement)?|Code Availability|"
               r"Availability of Data and Materials?|Supplementary (?:Information|Materials?))\s*$",
               re.IGNORECASE),
]

# Sections from which no dataset identifiers are expected.  Everything from
# the first matched heading onward is excluded from full_text to cut tokens.
_CUT_FROM_RE = re.compile(
    r"^(\d+(?:\.\d+)*\s+)?(references?|bibliography|acknowledg(?:e?ments?)|"
    r"author\s+contributions?|competing\s+interests?|conflict\s+of\s+interest|"
    r"ethics?\s+(statement|declaration)s?|funding)\s*$",
    re.IGNORECASE,
)


def _is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 100:
        return False
    for pat in HEADING_PATTERNS:
        if pat.match(line):
            return True
    # ALL CAPS short lines are likely headings
    if line.isupper() and 3 < len(line) < 60 and not any(c.isdigit() for c in line):
        return True
    return False


def _extract_pages(path: Path) -> list[tuple[int, str]]:
    fitz = _import_fitz()
    if fitz is not None:
        doc = fitz.open(str(path))
        try:
            return [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]
        finally:
            doc.close()
    pdfplumber = _import_pdfplumber()
    if pdfplumber is not None:
        with pdfplumber.open(str(path)) as pdf:
            return [(i + 1, (page.extract_text() or "")) for i, page in enumerate(pdf.pages)]
    raise RuntimeError(
        "Neither PyMuPDF nor pdfplumber is installed. Install one to parse PDF files."
    )


def parse_pdf_file(path: Path) -> dict[str, Any]:
    pages = _extract_pages(Path(path))
    # Keep raw page text for identifier (DOI/PMCID/PMID) regex extraction only.
    _raw_full_text = "\n".join(text for _, text in pages)

    sections: list[dict[str, Any]] = []
    current_title = ""
    current_buf: list[str] = []
    current_page: int | None = None
    path_idx = 0

    def flush() -> None:
        nonlocal current_buf
        text = "\n".join(current_buf).strip()
        if current_title or text:
            sections.append({
                "section_title": current_title,
                "section_text": text,
                "section_path": f"body/sec[{path_idx}]",
                "page": current_page,
            })
        current_buf = []

    for page_num, page_text in pages:
        for line in page_text.splitlines():
            if _is_heading(line):
                flush()
                path_idx += 1
                current_title = line.strip()
                current_page = page_num
            else:
                current_buf.append(line)
    flush()

    # Drop References and other non-informative tail sections to shrink
    # full_text.  We search forward to the *first* matching section and cut
    # there; earlier sections with the same name (very rare) are kept.
    cut_idx = len(sections)
    for i, s in enumerate(sections):
        if _CUT_FROM_RE.match(s["section_title"].strip()):
            cut_idx = i
            break
    sections = sections[:cut_idx]

    # Rebuild full_text from the filtered sections so it excludes the
    # reference list and other noise.  Fall back to raw pages only when
    # section parsing produced nothing at all.
    if sections:
        full_text = "\n".join(
            "\n".join(filter(None, [s["section_title"], s["section_text"]]))
            for s in sections
        ).strip()
    else:
        full_text = _raw_full_text

    # Title heuristic: first non-empty line of page 1
    title = ""
    if pages:
        for line in pages[0][1].splitlines():
            if line.strip():
                title = line.strip()
                break

    # Abstract heuristic: text in section titled Abstract
    abstract = ""
    for s in sections:
        if s["section_title"].lower().startswith("abstract"):
            abstract = s["section_text"]
            break

    doi_m = DOI_RE.search(_raw_full_text)
    pmcid_m = PMCID_RE.search(_raw_full_text)
    pmid_m = PMID_RE.search(_raw_full_text)
    doi = doi_m.group(0) if doi_m else ""
    pmcid = pmcid_m.group(0).upper() if pmcid_m else ""
    pmid = pmid_m.group(1) if pmid_m else ""
    paper_id = pmcid or pmid or doi or Path(path).stem

    return {
        "paper_id": paper_id,
        "paper_doi": doi,
        "pmcid": pmcid,
        "pmid": pmid,
        "source_path": str(path),
        "title": title,
        "abstract": abstract,
        "sections": sections,
        "full_text": full_text,
    }
