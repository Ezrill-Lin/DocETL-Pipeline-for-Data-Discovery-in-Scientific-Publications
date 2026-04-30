"""Build DocETL-ready JSONL input from a directory of HTML/PDF papers.

Each output record contains the structured paper plus a `candidate_passages`
field: a string concatenation of the most likely passages where dataset
references appear (data availability, methods, code availability,
supplementary, figure captions, plus any paragraph that mentions a known
repository or accession-like token). This keeps the LLM prompt focused
without requiring the model to ingest the entire paper.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .parse_html import parse_html_file
from .parse_pdf import parse_pdf_file

CANDIDATE_HEADING_RE = re.compile(
    r"(data\s+availability|availability\s+of\s+data|code\s+availability|"
    r"supplement|materials?\s+and\s+methods|methods?|results?|"
    r"experimental\s+procedures)",
    re.IGNORECASE,
)

# Strong identifier prefixes — presence signals a candidate passage
ACCESSION_RE = re.compile(
    r"\b(?:GSE|GSM|GPL|GDS|PXD|MSV|JPST|IPX|PASS|E-MTAB|E-GEOD|E-PROT|E-MEXP|"
    r"SRP|SRR|SRX|SRA|PRJNA|PRJEB|PRJDB|SAMN|SAMEA|SAMD|ERP|ERR|ERX|ERS)\d+\b"
)
REPO_NAME_RE = re.compile(
    r"\b(GEO|Gene Expression Omnibus|PRIDE|ProteomeXchange|MassIVE|jPOST|"
    r"iProX|PeptideAtlas|Panorama Public|ArrayExpress|SRA|BioProject|BioSample|"
    r"ENA|Dryad|Zenodo|Figshare|Dataverse|Mendeley Data|GenBank|UniProt)\b",
    re.IGNORECASE,
)
DOI_INLINE_RE = re.compile(r"\b10\.\d{4,9}/[\w\.\-/:;()]+", re.IGNORECASE)


def _looks_relevant(text: str) -> bool:
    if ACCESSION_RE.search(text):
        return True
    if REPO_NAME_RE.search(text):
        return True
    if "doi.org" in text.lower():
        return True
    if "available" in text.lower() and ("repository" in text.lower() or "deposited" in text.lower()):
        return True
    return False


def select_candidate_passages(paper: dict[str, Any], max_chars: int = 24_000) -> str:
    """Select passages most likely to contain dataset references."""
    chunks: list[str] = []
    seen: set[str] = set()

    def add(label: str, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        sig = text[:200]
        if sig in seen:
            return
        seen.add(sig)
        chunks.append(f"### {label}\n{text}")

    # Always include relevant-sounding sections by title
    for s in paper.get("sections", []):
        title = s.get("section_title", "") or ""
        body = s.get("section_text", "") or ""
        if CANDIDATE_HEADING_RE.search(title):
            add(title or "(section)", body)

    # Plus any section/paragraph containing accession-like or repo strings
    for s in paper.get("sections", []):
        body = s.get("section_text", "") or ""
        if not body:
            continue
        # Split into paragraphs to keep the prompt tight
        for para in re.split(r"\n{2,}", body):
            if _looks_relevant(para):
                add(s.get("section_title", "") or "(paragraph)", para)

    # Abstract sometimes carries DOIs / accessions
    if _looks_relevant(paper.get("abstract", "")):
        add("Abstract", paper["abstract"])

    text = "\n\n".join(chunks)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... truncated for prompt budget ...]"
    return text


def build_record(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in (".html", ".htm", ".xml", ".nxml"):
        paper = parse_html_file(path)
    elif suffix == ".pdf":
        paper = parse_pdf_file(path)
    else:
        raise ValueError(f"Unsupported file type for {path}")
    paper["candidate_passages"] = select_candidate_passages(paper)
    paper["candidate_char_count"] = len(paper["candidate_passages"])
    return paper


def iter_paper_files(input_dir: Path) -> Iterable[Path]:
    exts = {".html", ".htm", ".xml", ".nxml", ".pdf"}
    for p in sorted(Path(input_dir).rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def build_jsonl(input_dir: Path, output_path: Path) -> int:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output_path.open("w", encoding="utf-8") as f:
        for p in iter_paper_files(input_dir):
            try:
                record = build_record(p)
            except Exception as e:
                print(f"[WARN] Failed to parse {p}: {e}")
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def build_json_array(input_dir: Path, output_path: Path) -> int:
    """DocETL `type: file` accepts a JSON array. Convenience writer."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for p in iter_paper_files(input_dir):
        try:
            records.append(build_record(p))
        except Exception as e:
            print(f"[WARN] Failed to parse {p}: {e}")
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(records)
