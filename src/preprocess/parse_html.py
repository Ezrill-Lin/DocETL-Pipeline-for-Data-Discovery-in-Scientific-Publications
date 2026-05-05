"""Parse HTML / JATS XML scientific papers into structured records."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Comment, Tag

DOI_RE = re.compile(r"10\.\d{4,9}/[\w\.\-/:;()<>]+", re.IGNORECASE)
PMCID_RE = re.compile(r"PMC\d{4,}", re.IGNORECASE)
PMID_RE = re.compile(r"\bpmid[:\s]*(\d{4,9})\b", re.IGNORECASE)

# Tags that carry no readable content and inflate token counts.
_STRIP_TAGS: frozenset[str] = frozenset({
    # Browser / rendering noise
    "script", "style", "noscript",
    # Media embeds
    "img", "iframe", "video", "audio", "canvas", "svg",
    "object", "embed",
    # UI elements
    "button", "input", "select", "textarea", "form",
    # HTML document metadata (IDs are extracted from `raw` fallback)
    "head", "meta", "link",
})


def _strip_non_content(soup: BeautifulSoup) -> None:
    """Remove non-content elements in-place to reduce token count.

    Must be called after soup creation but before text extraction.
    ID-finder functions (_find_doi, _find_pmcid, _find_pmid) use the
    original `raw` string as a regex fallback, so stripping <meta> /
    <head> from the tree does not break metadata extraction.
    """
    for tag in soup.find_all(_STRIP_TAGS):
        tag.decompose()
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()


def _text(node: Tag | None) -> str:
    if node is None:
        return ""
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()


def _find_doi(soup: BeautifulSoup, fallback: str) -> str:
    # JATS pattern
    for tag in soup.find_all("article-id"):
        if tag.get("pub-id-type") == "doi":
            return _text(tag)
    # HTML meta
    for name in ("citation_doi", "DC.identifier", "dc.identifier"):
        meta = soup.find("meta", attrs={"name": name})
        if meta and meta.get("content"):
            return meta["content"].strip()
    m = DOI_RE.search(fallback)
    return m.group(0) if m else ""


def _find_pmcid(soup: BeautifulSoup, fallback: str) -> str:
    for tag in soup.find_all("article-id"):
        if tag.get("pub-id-type") == "pmcid":
            val = _text(tag)
            if not val.upper().startswith("PMC"):
                val = "PMC" + val
            return val
    m = PMCID_RE.search(fallback)
    return m.group(0).upper() if m else ""


def _find_pmid(soup: BeautifulSoup, fallback: str) -> str:
    for tag in soup.find_all("article-id"):
        if tag.get("pub-id-type") == "pmid":
            return _text(tag)
    meta = soup.find("meta", attrs={"name": "citation_pmid"})
    if meta and meta.get("content"):
        return meta["content"].strip()
    m = PMID_RE.search(fallback)
    return m.group(1) if m else ""


def _title(soup: BeautifulSoup) -> str:
    title = soup.find("article-title")
    if title:
        return _text(title)
    if soup.title:
        return _text(soup.title)
    h1 = soup.find("h1")
    return _text(h1)


def _abstract(soup: BeautifulSoup) -> str:
    abs_tag = soup.find("abstract")
    if abs_tag:
        return _text(abs_tag)
    meta = soup.find("meta", attrs={"name": "citation_abstract"})
    if meta and meta.get("content"):
        return meta["content"].strip()
    # heuristic: an element with class/id "abstract"
    candidate = soup.find(attrs={"class": re.compile(r"abstract", re.I)})
    if candidate:
        return _text(candidate)
    return ""


def _extract_jats_sections(soup: BeautifulSoup) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    body = soup.find("body")
    back = soup.find("back")
    front = soup.find("front")
    if not body and not back and not front:
        return sections

    # Some PMC papers (esp. older NIHMS-imported PLOS articles) put the
    # data-availability statement in <front><notes>. Capture those before
    # walking the body so the LLM never misses them to a content-budget cut.
    if front:
        for idx, note in enumerate(front.find_all("notes"), start=1):
            note_type = (note.get("notes-type") or "").strip()
            title_tag = note.find("title")
            title = _text(title_tag) if title_tag else (note_type or "Notes")
            text = "\n".join(_text(p) for p in note.find_all("p")) or _text(note)
            if text:
                sections.append({
                    "section_title": title,
                    "section_text": text,
                    "section_path": f"front/notes[{idx}]",
                    "page": None,
                })

    # Sections whose titles contain no dataset identifiers — skip entirely to
    # avoid inflating the full_text and FDR prompt with irrelevant content.
    _SKIP_SECTION_RE = re.compile(
        r"^(author\s+contributions?|competing\s+interests?|conflict\s+of\s+interest|"
        r"ethics?\s+(statement|declaration)s?|funding|declaration\s+of\s+competing)\b",
        re.IGNORECASE,
    )

    def walk(sec: Tag, path_parts: list[str]) -> None:
        title_tag = sec.find("title", recursive=False)
        title = _text(title_tag) if title_tag else ""
        if title and _SKIP_SECTION_RE.match(title.strip()):
            return
        # collect direct paragraph text plus captions
        text_parts: list[str] = []
        for child in sec.children:
            if not isinstance(child, Tag):
                continue
            if child.name in ("p", "list", "disp-quote", "boxed-text", "statement"):
                text_parts.append(_text(child))
            elif child.name in ("fig", "table-wrap", "supplementary-material"):
                cap = child.find("caption")
                if cap:
                    text_parts.append("[CAPTION] " + _text(cap))
        section_text = "\n".join(t for t in text_parts if t)
        if title or section_text:
            sections.append({
                "section_title": title,
                "section_text": section_text,
                "section_path": "/".join(path_parts) or "body/sec",
                "page": None,
            })
        for idx, sub in enumerate(sec.find_all("sec", recursive=False), start=1):
            walk(sub, path_parts + [f"sec[{idx}]"])

    if body:
        for idx, sec in enumerate(body.find_all("sec", recursive=False), start=1):
            walk(sec, [f"body/sec[{idx}]"])

        # Figure captions and supplementary material outside numbered sections
        for fig in body.find_all(["fig", "table-wrap", "supplementary-material"]):
            cap = fig.find("caption")
            if cap and _text(cap):
                sections.append({
                    "section_title": f"[{fig.name}]",
                    "section_text": _text(cap),
                    "section_path": fig.name,
                    "page": None,
                })

    # PMC JATS often files data-availability statements under
    # <back><notes notes-type="data-availability"> rather than as a body
    # <sec>. Harvest those plus any other notes/sec inside <back>.
    # Footnotes (<fn>) frequently carry the accession in older NIHMS papers.
    if back:
        # Reference lists are pure bibliography — drop them before any walking
        # so they never end up in full_text or sections.
        for ref_list in back.find_all("ref-list"):
            ref_list.decompose()

        for idx, note in enumerate(back.find_all("notes"), start=1):
            note_type = (note.get("notes-type") or "").strip()
            title_tag = note.find("title")
            title = _text(title_tag) if title_tag else (note_type or "Notes")
            text = "\n".join(_text(p) for p in note.find_all("p")) or _text(note)
            if text:
                sections.append({
                    "section_title": title,
                    "section_text": text,
                    "section_path": f"back/notes[{idx}]",
                    "page": None,
                })
        for idx, sec in enumerate(back.find_all("sec", recursive=False), start=1):
            walk(sec, [f"back/sec[{idx}]"])
        for idx, fn in enumerate(back.find_all("fn"), start=1):
            text = _text(fn)
            if text:
                sections.append({
                    "section_title": "[footnote]",
                    "section_text": text,
                    "section_path": f"back/fn[{idx}]",
                    "page": None,
                })

    return sections


def _extract_html_sections(soup: BeautifulSoup) -> list[dict[str, Any]]:
    """Heuristic section extraction for non-JATS HTML."""
    sections: list[dict[str, Any]] = []
    main = soup.find("main") or soup.find("article") or soup.body or soup
    current_title = ""
    current_buf: list[str] = []
    path_idx = 0

    def flush() -> None:
        nonlocal current_buf
        text = "\n".join(t for t in current_buf if t).strip()
        if current_title or text:
            sections.append({
                "section_title": current_title,
                "section_text": text,
                "section_path": f"body/sec[{path_idx}]",
                "page": None,
            })
        current_buf = []

    for el in main.descendants:
        if not isinstance(el, Tag):
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            flush()
            path_idx += 1
            current_title = _text(el)
        elif el.name in ("p", "li", "blockquote"):
            t = _text(el)
            if t:
                current_buf.append(t)
        elif el.name in ("figcaption",):
            t = _text(el)
            if t:
                current_buf.append("[CAPTION] " + t)
    flush()
    return sections


def parse_html_text(raw: str, source: str = "", default_paper_id: str = "") -> dict[str, Any]:
    """Parse a raw HTML or JATS XML *string* into a structured paper record.

    Args:
        raw: The HTML/XML source text.
        source: Free-form provenance string written to ``source_path`` (a file
            path, URL, or anything else useful for debugging).
        default_paper_id: Used when no PMCID / PMID / DOI is found.
    """
    # Try lxml-xml for JATS, fall back to lxml HTML
    is_xml = "<article" in raw[:2000] or raw.lstrip().startswith("<?xml")
    soup = BeautifulSoup(raw, "lxml-xml") if is_xml else BeautifulSoup(raw, "lxml")
    _strip_non_content(soup)


    article_node = soup.find("article")
    if article_node and article_node.find("body"):
        sections = _extract_jats_sections(soup)
    else:
        sections = _extract_html_sections(soup)

    title = _title(soup)
    abstract = _abstract(soup)
    full_text_parts = [title, abstract]
    for s in sections:
        if s["section_title"]:
            full_text_parts.append(s["section_title"])
        if s["section_text"]:
            full_text_parts.append(s["section_text"])
    full_text = "\n".join(p for p in full_text_parts if p)

    pmcid = _find_pmcid(soup, raw)
    pmid = _find_pmid(soup, raw)
    doi = _find_doi(soup, raw)

    paper_id = pmcid or pmid or doi or default_paper_id

    return {
        "paper_id": paper_id,
        "paper_doi": doi,
        "pmcid": pmcid,
        "pmid": pmid,
        "source_path": source,
        "title": title,
        "abstract": abstract,
        "sections": sections,
        "full_text": full_text,
    }


def parse_html_file(path: Path) -> dict[str, Any]:
    """Parse an HTML or JATS XML file into a structured paper record."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="ignore")
    return parse_html_text(raw, source=str(path), default_paper_id=path.stem)
