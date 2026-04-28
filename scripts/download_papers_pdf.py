"""
Download PMC paper HTML and PDF files using Playwright + Edge.

This script uses a real browser session to let PMC set its proof-of-work
cookie, then performs the final PDF download with regular HTTP requests.
"""
from __future__ import annotations

import csv
import os
import re
import time
from pathlib import Path
from typing import Dict, List
from urllib.parse import urljoin

import requests
from playwright.sync_api import sync_playwright


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CSV_PATH = REPO_ROOT / "data" / "groundtruth" / "EXP_groundtruth.csv"
OUTPUT_ROOT = REPO_ROOT / "data" / "input"
PDF_DIR = OUTPUT_ROOT / "pdf"
HTML_DIR = OUTPUT_ROOT / "html"
RESULTS_PATH = OUTPUT_ROOT / "download_results_pdf.csv"
EDGE_PATH = Path(
    os.getenv(
        "EDGE_PATH",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    )
)
USER_DATA_DIR = SCRIPT_DIR / ".edge-profile"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36 Edg/136.0.0.0"
)
HEADLESS = os.getenv("HEADLESS") == "1"
MAX_ARTICLES = int(os.getenv("MAX_ARTICLES", "0") or "0")
INCLUDE_ARTICLES = {
    value.strip().upper()
    for value in os.getenv("INCLUDE_ARTICLES", "").split(",")
    if value.strip()
}


def article_id_from_url(url: str) -> str:
    match = re.search(r"(PMC\d+)", url, re.IGNORECASE)
    if not match:
        raise ValueError(f"Unable to derive PMC id from URL: {url}")
    return match.group(1).upper()


def meta_content(html: str, name: str) -> str:
    pattern = re.compile(
        rf'<meta\s+name="{re.escape(name)}"\s+content="([^"]+)"',
        re.IGNORECASE,
    )
    match = pattern.search(html)
    return match.group(1) if match else ""


def read_urls() -> List[str]:
    unique_urls: List[str] = []
    seen = set()

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            url = (row.get("citing_publication_link") or "").strip()
            if url and url not in seen:
                seen.add(url)
                unique_urls.append(url)

    return unique_urls


def resolve_pdf_url(article_id: str, html: str) -> str:
    referer = f"https://pmc.ncbi.nlm.nih.gov/articles/{article_id}/"
    tagged_pdf_url = meta_content(html, "citation_pdf_url")
    if tagged_pdf_url:
        return urljoin(referer, tagged_pdf_url)
    return urljoin(referer, "./pdf/")


def get_pow_cookies(context, page, pdf_url: str) -> List[Dict[str, str]]:
    timeout_at = time.time() + 45
    try:
        page.goto(pdf_url, wait_until="domcontentloaded", timeout=45000)
    except Exception:
        pass

    while time.time() < timeout_at:
        cookies = context.cookies(pdf_url)
        pow_cookie = next(
            (cookie for cookie in cookies if cookie.get("name") == "cloudpmc-viewer-pow"),
            None,
        )
        if pow_cookie and pow_cookie.get("value"):
            return cookies
        time.sleep(1)

    raise RuntimeError(f"Timed out waiting for PMC proof-of-work cookie for {pdf_url}")


def build_cookie_header(cookies: List[Dict[str, str]]) -> str:
    return "; ".join(f"{cookie['name']}={cookie['value']}" for cookie in cookies)


def download_pdf(pdf_url: str, referer: str, cookie_header: str, pdf_path: Path) -> None:
    response = requests.get(
        pdf_url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
            "Referer": referer,
            "Cookie": cookie_header,
        },
        timeout=120,
    )
    response.raise_for_status()

    if not response.content.startswith(b"%PDF-"):
        preview = response.content[:240].decode("utf-8", errors="replace")
        preview = " ".join(preview.split())
        raise RuntimeError(f"PDF endpoint returned non-PDF content: {preview[:160]}")

    pdf_path.write_bytes(response.content)


def save_results(results: List[Dict[str, str]]) -> None:
    fieldnames = ["article_id", "source_url", "pdf_path", "html_path", "status", "error"]
    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    if not EDGE_PATH.exists():
        raise FileNotFoundError(f"Edge not found at {EDGE_PATH}")

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

    worklist = read_urls()
    if INCLUDE_ARTICLES:
        worklist = [url for url in worklist if article_id_from_url(url) in INCLUDE_ARTICLES]
    if MAX_ARTICLES > 0:
        worklist = worklist[:MAX_ARTICLES]

    results: List[Dict[str, str]] = []

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=str(USER_DATA_DIR),
            executable_path=str(EDGE_PATH),
            headless=HEADLESS,
            accept_downloads=False,
            viewport={"width": 1440, "height": 900},
            user_agent=USER_AGENT,
            args=["--disable-blink-features=AutomationControlled"],
        )

        page = context.pages[0] if context.pages else context.new_page()
        page.set_default_timeout(45000)

        try:
            for source_url in worklist:
                article_id = article_id_from_url(source_url)
                html_path = HTML_DIR / f"{article_id}.html"
                pdf_path = PDF_DIR / f"{article_id}.pdf"
                referer = f"https://pmc.ncbi.nlm.nih.gov/articles/{article_id}/"

                try:
                    print(f"Downloading {article_id}")
                    page.goto(source_url, wait_until="domcontentloaded", timeout=45000)
                    try:
                        page.wait_for_load_state("networkidle", timeout=45000)
                    except Exception:
                        pass

                    html = page.content()
                    html_path.write_text(html, encoding="utf-8")

                    pdf_url = resolve_pdf_url(article_id, html)
                    cookies = get_pow_cookies(context, page, pdf_url)
                    cookie_header = build_cookie_header(cookies)
                    download_pdf(pdf_url, referer, cookie_header, pdf_path)

                    results.append(
                        {
                            "article_id": article_id,
                            "source_url": source_url,
                            "pdf_path": str(pdf_path),
                            "html_path": str(html_path),
                            "status": "ok",
                            "error": "",
                        }
                    )
                except Exception as error:
                    if pdf_path.exists():
                        pdf_path.unlink()
                    results.append(
                        {
                            "article_id": article_id,
                            "source_url": source_url,
                            "pdf_path": str(pdf_path),
                            "html_path": str(html_path),
                            "status": "failed",
                            "error": str(error),
                        }
                    )
                    print(f"Failed {article_id}: {error}")
        finally:
            save_results(results)
            context.close()

    success_count = sum(1 for result in results if result["status"] == "ok")
    failure_count = len(results) - success_count
    print(f"ok: {success_count}")
    print(f"failed: {failure_count}")


if __name__ == "__main__":
    main()
