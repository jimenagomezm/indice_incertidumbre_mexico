"""Shared web-scraping mechanics, reused by every document source.

Nothing here knows anything about a particular Banxico page. What *is*
source-specific - which URL to read, how its HTML is laid out, how to parse a
title into a date - lives in that source's own module, e.g.
`corpus/banxico_anuncios.py`.

The split is deliberately drawn here and not further. It would be possible to
turn the scraper itself into one configurable function taking the link text, the
row structure, the date format and the encoding as arguments. That was
considered and rejected: those are not four independent settings, they are one
page's layout observed once, and a function with eight such parameters is harder
to read than two short functions that each mirror the page they scrape. A new
source gets a new module.
"""

import time
from pathlib import Path

import pandas as pd
import requests
from lxml import html
from tqdm.auto import tqdm

HEADERS = {
    "User-Agent": (
        "MexicoUncertaintyIndex/0.1 "
        "(academic research; jimenagm98@hotmail.com)"
    ),
    "Accept-Language": "es-MX,es;q=0.9,en;q=0.8",
}



def clean_text(value):
    """Collapse repeated whitespace/line breaks in short metadata fields."""
    if value is None:
        return None
    return " ".join(str(value).split()).strip()


def fetch_page_html(url, headers=None, encoding=None, timeout=60):
    """Download a page and return it as a parsed HTML tree.

    `encoding` is a parameter because it genuinely varies by site, and getting
    it wrong is a silent failure: the request succeeds and every accented
    character comes back corrupted. Pass the encoding the server actually uses;
    leave it None to let requests guess.
    """
    if headers is None:
        headers = HEADERS

    response = requests.get(url, headers=headers, timeout=timeout)
    # Abort on an HTTP error instead of parsing an error page as if it were data.
    response.raise_for_status()

    if encoding is None:
        markup = response.text
    else:
        markup = response.content.decode(encoding)

    return html.fromstring(markup, base_url=url)


def download_pdfs(metadata, pdf_dir, headers=HEADERS):
    """Download each report PDF listed in `metadata` into `pdf_dir`.

    Existing files are not re-downloaded. Returns a dataframe recording
    where each file was saved and whether the download succeeded.
    """
    results = []
    # A Session reuses one connection across all downloads (faster) and
    # applies our polite headers to every request.
    session = requests.Session()
    session.headers.update(headers)

    for row in tqdm(metadata.itertuples(index=False), total=len(metadata)):
        pdf_path = pdf_dir / f"{row.document_id}.pdf"

        # Skip anything already on disk so re-running is cheap and idempotent.
        if pdf_path.exists():
            results.append(
                {
                    "document_id": row.document_id,
                    "local_pdf_path": str(pdf_path),
                    "download_status": "already_exists",
                }
            )
            continue

        try:
            response = session.get(row.pdf_url, timeout=120)
            response.raise_for_status()

            # Make sure we actually got a PDF, not an HTML error page dressed
            # up as one: trust either the Content-Type header or the magic
            # bytes "%PDF" at the very start of the file.
            content_type = response.headers.get("Content-Type", "").lower()
            is_pdf = "pdf" in content_type or response.content[:4] == b"%PDF"

            if not is_pdf:
                results.append(
                    {
                        "document_id": row.document_id,
                        "local_pdf_path": None,
                        "download_status": f"not_pdf: {content_type}",
                    }
                )
                continue

            # Save the raw PDF bytes to disk.
            pdf_path.write_bytes(response.content)

            results.append(
                {
                    "document_id": row.document_id,
                    "local_pdf_path": str(pdf_path),
                    "download_status": "downloaded",
                }
            )

            # small pause to avoid sending requests too quickly
            time.sleep(0.5)

        except requests.RequestException as error:
            # Network/HTTP failure for this one file: record it and keep going
            # with the rest instead of aborting the whole batch.
            results.append(
                {
                    "document_id": row.document_id,
                    "local_pdf_path": None,
                    "download_status": f"error: {error}",
                }
            )

    return pd.DataFrame(results)
