"""Download PDF papers via browser automation (Playwright).

Use this when a paper URL is not API-accessible (e.g. journal publisher pages
that require a browser session, DOI redirects, etc.).  The downloaded PDF is
saved to ``<raw_dir>/<paper_id>.pdf``.

Requires:
    playwright>=1.40 (``uv add playwright``)
    playwright chromium installed: ``uv run playwright install chromium``

Typical usage
-------------
    from src.preprocess.browser_automation import download_pdf

    path = download_pdf(
        url="https://doi.org/10.1038/s41586-021-03819-2",
        paper_id="s41586-021-03819-2",
        raw_dir=Path("data/raw"),
    )
    print(path)  # data/raw/s41586-021-03819-2.pdf

Or as a CLI::

    uv run python -m src.preprocess.browser_automation \\
        --url https://doi.org/10.1038/s41586-021-03819-2 \\
        --paper-id s41586-021-03819-2 \\
        --raw-dir data/raw

Multiple URLs (TSV: paper_id<TAB>url, one per line)::

    uv run python -m src.preprocess.browser_automation \\
        --urls-tsv urls.tsv \\
        --raw-dir data/raw
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterator


class BrowserAutomationError(RuntimeError):
    pass


def _require_playwright():
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except ImportError as exc:
        raise BrowserAutomationError(
            "playwright is not installed. Run: uv add playwright && "
            "uv run playwright install chromium"
        ) from exc


def _sanitize_paper_id(paper_id: str) -> str:
    """Replace characters that are invalid in filenames."""
    invalid = r'\/:*?"<>|'
    for ch in invalid:
        paper_id = paper_id.replace(ch, "_")
    return paper_id


def download_pdf(
    url: str,
    paper_id: str,
    raw_dir: Path,
    *,
    timeout_ms: int = 60_000,
    wait_for_pdf_ms: int = 10_000,
    headless: bool = True,
    overwrite: bool = False,
) -> Path:
    """Navigate to *url* in a Chromium browser, find and download the PDF.

    Strategy
    --------
    1. Open the page and intercept any response whose ``Content-Type`` is
       ``application/pdf``.  Many publishers serve the PDF inline in an
       ``<iframe>`` or redirect to it after JavaScript runs.
    2. If no PDF response is intercepted within *wait_for_pdf_ms*, look for
       a ``<a>`` / ``<button>`` whose text or ``href`` signals a PDF link
       (e.g. "Download PDF", "Full Text PDF") and click it.
    3. As a last resort, capture the page print-to-PDF (works for
       HTML-rendered full-texts but produces a scanned-style PDF).

    The file is written atomically: data is first written to
    ``<paper_id>.pdf.tmp`` then renamed.

    Parameters
    ----------
    url:
        Landing page URL (DOI link, journal article page, etc.).
    paper_id:
        Used as the output filename stem, e.g. ``PMC12345678`` or a DOI
        fragment.  Any filesystem-illegal characters are replaced with ``_``.
    raw_dir:
        Directory where the PDF will be saved.  Created if absent.
    timeout_ms:
        Playwright navigation timeout in milliseconds.
    wait_for_pdf_ms:
        How long to wait for a PDF response after clicking a PDF link.
    headless:
        Run Chromium in headless mode.  Set ``False`` to debug visually.
    overwrite:
        If ``False`` (default) and the target file already exists, skip the
        download and return the existing path.

    Returns
    -------
    Path
        Absolute path to the saved PDF file.
    """
    _require_playwright()
    from playwright.sync_api import sync_playwright, Response, Download

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    safe_id = _sanitize_paper_id(paper_id)
    dest = raw_dir / f"{safe_id}.pdf"
    tmp = raw_dir / f"{safe_id}.pdf.tmp"

    if dest.exists() and not overwrite:
        return dest.resolve()

    pdf_bytes: list[bytes] = []

    def _on_response(response: Response) -> None:
        ct = response.headers.get("content-type", "")
        if "application/pdf" in ct:
            try:
                pdf_bytes.append(response.body())
            except Exception:
                pass

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        page.on("response", _on_response)

        try:
            page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
        except Exception as exc:
            browser.close()
            raise BrowserAutomationError(f"Navigation failed for {url}: {exc}") from exc

        # Give JavaScript a moment to fire redirects / load iframes
        page.wait_for_timeout(2000)

        # -- Strategy 1: intercepted PDF response ----------------------------
        if pdf_bytes:
            tmp.write_bytes(pdf_bytes[0])
            tmp.rename(dest)
            browser.close()
            return dest.resolve()

        # -- Strategy 2: click a PDF download link ---------------------------
        pdf_link_selectors = [
            "a[href$='.pdf']",
            "a[href*='/pdf']",
            "a[href*='pdf=']",
            "a:text-matches('(Download|View|Full.?Text).*PDF', 'i')",
            "button:text-matches('(Download|View|Full.?Text).*PDF', 'i')",
        ]

        download_result: Download | None = None
        for selector in pdf_link_selectors:
            element = page.query_selector(selector)
            if element is None:
                continue
            try:
                with context.expect_download(timeout=wait_for_pdf_ms) as dl_info:
                    element.click()
                download_result = dl_info.value
                break
            except Exception:
                # Selector matched but click didn't trigger a download —
                # try next selector.
                pass

        if download_result is not None:
            download_result.save_as(str(tmp))
            tmp.rename(dest)
            browser.close()
            return dest.resolve()

        # Check again: the click may have triggered a PDF response instead of
        # a download dialog.
        page.wait_for_timeout(wait_for_pdf_ms)
        if pdf_bytes:
            tmp.write_bytes(pdf_bytes[0])
            tmp.rename(dest)
            browser.close()
            return dest.resolve()

        # -- Strategy 3: print-to-PDF fallback --------------------------------
        # Works for HTML full-texts (e.g. PMC reading room, preprint servers).
        pdf_data = page.pdf(format="A4", print_background=True)
        browser.close()

        if not pdf_data:
            raise BrowserAutomationError(
                f"All download strategies failed for {url}"
            )

        tmp.write_bytes(pdf_data)
        tmp.rename(dest)
        return dest.resolve()


def iter_tsv(path: Path) -> Iterator[tuple[str, str]]:
    """Yield ``(paper_id, url)`` pairs from a two-column TSV file.

    Lines starting with ``#`` and blank lines are skipped.
    """
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        yield parts[0].strip(), parts[1].strip()


def download_many(
    entries: list[tuple[str, str]],
    raw_dir: Path,
    *,
    delay_s: float = 1.5,
    headless: bool = True,
    overwrite: bool = False,
) -> dict[str, Path | Exception]:
    """Download multiple PDFs sequentially with a polite inter-request delay.

    Parameters
    ----------
    entries:
        List of ``(paper_id, url)`` pairs.
    raw_dir:
        Output directory.
    delay_s:
        Seconds to wait between requests (be a good citizen).
    headless:
        Passed through to :func:`download_pdf`.
    overwrite:
        Passed through to :func:`download_pdf`.

    Returns
    -------
    dict
        Maps each *paper_id* to the saved :class:`~pathlib.Path` on success,
        or the :class:`Exception` on failure.
    """
    results: dict[str, Path | Exception] = {}
    for i, (paper_id, url) in enumerate(entries):
        if i > 0:
            time.sleep(delay_s)
        try:
            path = download_pdf(
                url=url,
                paper_id=paper_id,
                raw_dir=raw_dir,
                headless=headless,
                overwrite=overwrite,
            )
            results[paper_id] = path
            print(f"[{i+1}/{len(entries)}] OK  {paper_id} -> {path}")
        except Exception as exc:
            results[paper_id] = exc
            print(f"[{i+1}/{len(entries)}] ERR {paper_id}: {exc}", file=sys.stderr)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.preprocess.browser_automation",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--url", help="Single paper URL to download.")
    mode.add_argument(
        "--urls-tsv",
        type=Path,
        metavar="FILE",
        help="TSV file with two columns: paper_id<TAB>url (one per line).",
    )
    p.add_argument(
        "--paper-id",
        help="Paper ID / output filename stem (required with --url).",
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to save PDFs (default: data/raw).",
    )
    p.add_argument(
        "--no-headless",
        action="store_true",
        help="Show the browser window (useful for debugging).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download even if the file already exists.",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=1.5,
        metavar="SECONDS",
        help="Delay between downloads when using --urls-tsv (default: 1.5s).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    headless = not args.no_headless

    if args.url:
        if not args.paper_id:
            parser.error("--paper-id is required when using --url")
        try:
            path = download_pdf(
                url=args.url,
                paper_id=args.paper_id,
                raw_dir=args.raw_dir,
                headless=headless,
                overwrite=args.overwrite,
            )
            print(f"Saved: {path}")
        except BrowserAutomationError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    else:  # --urls-tsv
        entries = list(iter_tsv(args.urls_tsv))
        if not entries:
            print("No entries found in TSV file.", file=sys.stderr)
            return 1
        results = download_many(
            entries,
            raw_dir=args.raw_dir,
            delay_s=args.delay,
            headless=headless,
            overwrite=args.overwrite,
        )
        errors = {pid: exc for pid, exc in results.items() if isinstance(exc, Exception)}
        print(f"\nDone: {len(results) - len(errors)} succeeded, {len(errors)} failed.")
        if errors:
            print("Failed paper IDs:", ", ".join(errors.keys()), file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
