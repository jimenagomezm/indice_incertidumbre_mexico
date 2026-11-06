"""Banco de México monetary policy announcements (*Anuncios de Política Monetaria*).

Everything specific to this one publication: where its index page is, how a
title like "El objetivo para la Tasa de Interés Interbancaria ... disminuye en 25
puntos base" turns into a policy decision, and how document ids are formed. The
generic machinery it builds on - fetching a page, downloading PDFs - lives in
`corpus/scraping.py`. Every source returns metadata with at least these columns:
document_id, institution, document_type, title, publication_date,
publication_month, reference_year, quarter, pdf_url, source_page_url.

**How this source differs from the quarterly reports.** An announcement is
dated by its decision day, not by a reference quarter, so `reference_year` and
`quarter` are derived from the publication date. Several announcements fall in
one quarter (about two since 2008), which is why the id carries the full date.

**The instrument changed in 2008.** Until January 2008 Banxico announced
changes to the "corto" (a target for commercial banks' current-account
balances); since then it announces a target for the overnight interbank rate.
Before 2003 there was no fixed calendar, so 2000-2002 only contains the
announcements where the corto changed. `policy_instrument` records which regime
a document belongs to, so the index stage can decide what to include.

Extra columns on top of the shared contract: `decision` (tighten / ease /
hold, read from the title) and `policy_instrument` (corto / tasa_objetivo).
"""

import re
from urllib.parse import urljoin

import pandas as pd

from scraping import HEADERS, clean_text, fetch_page_html

# Prefix for this source's document ids, e.g. banxico_apm_2026_09_24.
DOCUMENT_ID_PREFIX = "banxico_apm"

PAGE_URL = (
    "https://www.banxico.org.mx/"
    "publicaciones-y-prensa/anuncios-de-las-decisiones-de-politica-monetaria/"
    "anuncios-politica-monetaria-t.html"
)

# Title wording that marks each kind of decision, checked in this order. Order
# matters: "Relajamiento de las condiciones monetarias ... El 'corto' se
# mantiene sin cambio" is an easing, even though it also says "sin cambio".
DECISION_PATTERNS = [
    ("tighten", r"se incrementa|se aumenta|se restringen"),
    ("ease", r"disminuye|se reduce|relajamiento"),
    ("hold", r"sin cambio"),
]


def parse_decision(title):
    """Classify a title as 'tighten', 'ease' or 'hold' (None if unrecognized)."""
    if not title:
        return None
    lowered = title.lower()
    for decision, pattern in DECISION_PATTERNS:
        if re.search(pattern, lowered):
            return decision
    return None


def parse_policy_instrument(title):
    """Which instrument the announcement is about: the corto or the target rate."""
    if not title:
        return None
    if "tasa objetivo" in title.lower() or "tasa de interés" in title.lower():
        return "tasa_objetivo"
    if "corto" in title.lower():
        return "corto"
    return None


# Since 2021 every announcement ends with a table of inflation forecasts. The
# PDF noise filter keeps it, because a column like "Actual - INPC1/ 3.3 4.3 ..."
# has just enough letters to pass as text, so it reaches the index as dozens of
# lines of loose numbers. The table always starts with this heading and runs to
# the end of the document, footnotes included.
FORECAST_TABLE_HEADING = re.compile(
    r"pron[oó]sticos de la inflaci[oó]n general y subyacente", re.IGNORECASE)


def strip_forecast_table(page_text):
    """Drop the forecast table: everything from its heading to the end of the page.

    Applied page by page. The table sits on the last page, after the closing
    prose, so nothing an index should read is cut.
    """
    match = FORECAST_TABLE_HEADING.search(page_text)
    if match is None:
        return page_text
    return page_text[:match.start()].rstrip()


def build_document_id(row):
    """The stable id for one announcement, e.g. banxico_apm_2026_09_24.

    The positional fallback is for a row whose date could not be parsed; it is
    deliberately ugly, because producing one means the page's date format
    changed and the parser needs updating.
    """
    if pd.notna(row["publication_date"]):
        return f"{DOCUMENT_ID_PREFIX}_{row['publication_date']:%Y_%m_%d}"
    return f"{DOCUMENT_ID_PREFIX}_{row.name:03d}"


def scrape_index(page_url=PAGE_URL, headers=HEADERS):
    """Scrape Banxico's monetary policy announcement index metadata.

    Returns one row per announcement (a catalog of documents, not their text).
    """
    # Served as latin-1, like the quarterly report page. Getting that wrong
    # does not raise - it silently mangles every accented character.
    tree = fetch_page_html(page_url, headers=headers, encoding="latin-1")

    # Every announcement's PDF link has the visible text "Texto completo".
    # This also skips the one other PDF on the page, the decision calendar.
    full_text_links = tree.xpath(
        "//a[contains(normalize-space(string(.)), 'Texto completo')]"
    )

    records = []
    for link in full_text_links:
        href = link.get("href")
        if not href:
            continue

        # Walk up from the link to its table row: date cell, then title cell.
        row_matches = link.xpath("ancestor::tr[1]")
        if not row_matches:
            continue

        cells = row_matches[0].xpath("./td | ./th")
        if len(cells) < 2:
            continue

        date_text = clean_text(cells[0].text_content())
        resource_text = clean_text(cells[1].text_content())

        # The title is everything before the "Texto completo" link text.
        title = resource_text.split("Texto completo", 1)[0].strip(" |")

        # Publication date, format dd/mm/yy.
        date_match = re.search(r"\b\d{2}/\d{2}/\d{2}\b", date_text)
        publication_date = date_match.group(0) if date_match else None

        records.append(
            {
                "publication_date_raw": publication_date,
                "title": title,
                "resource_type": "Texto completo",
                "pdf_url": urljoin(page_url, href),
                "source_page_url": page_url,
                "institution": "Banco de México",
                "document_type": "Anuncio de política monetaria",
            }
        )

    metadata = pd.DataFrame(records)
    if metadata.empty:
        return metadata

    metadata = metadata.drop_duplicates(subset=["pdf_url"]).reset_index(drop=True)

    metadata["publication_date"] = pd.to_datetime(
        metadata["publication_date_raw"],
        format="%d/%m/%y",
        errors="coerce",
    )

    # The quarter an announcement belongs to is the quarter it was made in.
    metadata["reference_year"] = metadata["publication_date"].dt.year
    metadata["quarter"] = metadata["publication_date"].dt.quarter
    metadata["publication_month"] = metadata["publication_date"].dt.to_period("M").astype("string")

    # Every announcement is the main document; the column exists because the
    # shared contract has it (the quarterly reports have addenda).
    metadata["document_variant"] = "announcement"

    metadata["decision"] = metadata["title"].apply(parse_decision)
    metadata["policy_instrument"] = metadata["title"].apply(parse_policy_instrument)

    metadata["document_id"] = metadata.apply(build_document_id, axis=1)

    columns = [
        "document_id",
        "institution",
        "document_type",
        "document_variant",
        "title",
        "publication_date",
        "publication_month",
        "reference_year",
        "quarter",
        "decision",
        "policy_instrument",
        "resource_type",
        "pdf_url",
        "source_page_url",
    ]

    return metadata[columns]
