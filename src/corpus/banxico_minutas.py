"""Banco de México board minutes (*Minutas*) and meeting transcripts (*Transcripciones*).

Both are listed on the same page, one row per monetary policy decision, so one
module scrapes both. They are nevertheless two corpora - kept apart by
`document_type` and by id prefix - because they are different kinds of text:

- **Minuta** (2011 onward, every scheduled decision): an edited summary of the
  board's discussion, with views attributed by count ("la mayoría", "algunos",
  "uno", "otro") rather than by name.
- **Transcripción** (only meetings from 2018 to 2023): each board member's
  statement verbatim, under their name. The closest Banxico text to the
  speaker-level design of PMU (Cieslak, Hansen, McMahon & Xiao).

Every source returns metadata with at least these columns:
document_id, institution, document_type, title, publication_date,
publication_month, reference_year, quarter, pdf_url, source_page_url.

**Dated by the decision, not by publication.** A minuta comes out about two
weeks after the meeting, and a transcript years later; both are about the
decision announced on the date in their title. So `reference_year` and
`quarter` come from that decision date, and the id carries it:
`banxico_min_2023_08_10` and `banxico_tr_2023_08_10` discuss the decision
announced in `banxico_apm_2023_08_10`. `publication_date` keeps the real
publication date.

**Only the board's discussion is kept.** A minuta also carries staff's
presentation of economic conditions - not the board's view, full of chart
labels, and overlapping the quarterly reports. Where it sits changed over time:
until early 2018 it is section 2, "EVOLUCIÓN ECONÓMICA Y FINANCIERA", *before*
the board's discussion; from mid-2018 it is an ANEXO at the end. Both are
dropped by `keep_board_discussion`, which keeps the text from the heading
"ANÁLISIS Y MOTIVACIÓN DE LOS VOTOS DE LOS MIEMBROS DE LA JUNTA DE GOBIERNO" up
to the annex: the discussion, the decision, the vote and any dissent. The cover
page and the list of attendees go too.
"""

import re
from urllib.parse import urljoin

import pandas as pd

from scraping import HEADERS, clean_text, fetch_page_html

PAGE_URL = (
    "https://www.banxico.org.mx/"
    "publicaciones-y-prensa/minutas-de-las-decisiones-de-politica-monetaria/"
    "minutas-politica-monetaria-ta.html"
)

# Visible link text on the page -> (document_type, id prefix).
DOCUMENT_KINDS = {
    "Minuta": ("Minuta", "banxico_min"),
    "Transcripción": ("Transcripción", "banxico_tr"),
}

SPANISH_MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11,
    "diciembre": 12,
}


# Passed to the PDF extractor as `column_options={"noise_markers": ...}`. The
# default drops any block that *starts* with "Fuente", "Nota", "Gráfica"...;
# in these PDFs a chart's source line is often merged into one block with the
# paragraph after it, so the default threw away real prose - including, in
# 2011-2012, the very heading `keep_board_discussion` looks for. This version
# drops such a block only when it is the caption alone, i.e. nothing follows it
# after a blank line. Recovers ~7% of the minutas' words.
CAPTION_ONLY_MARKERS = re.compile(
    r"^\s*(gr[aá]fica|cuadro|tabla|figura|fuente|nota)\b(?![\s\S]*\n[ \t]*\n\s*\S)",
    re.IGNORECASE)

BOARD_SECTION_START = re.compile(
    r"^\s*\d\.\s*AN[ÁA]LISIS Y MOTIVACI[ÓO]N DE LOS VOTOS", re.MULTILINE)
# Since 2024 the heading shares its line with the first sentence
# ("ANEXO La información presentada..."), so this is not anchored at line end.
ANNEX_START = re.compile(r"^\s*ANEXO\b", re.MULTILINE)


def keep_board_discussion(pages):
    """Cut every minuta down to the board's own discussion.

    Works on the page table (one row per page) and keeps its shape: a page
    entirely outside the discussion is left with empty text, which
    `flag_page_quality` then marks unusable, so it never reaches the chunks. A
    document where the start heading is never found ends up empty - that is
    how an unreadable PDF shows itself (see the notebook).

    Also repairs the "ti" ligature that some 2018-2024 PDFs extract as "Ɵ".
    """
    pages = pages.sort_values(["document_id", "page_number"]).copy()
    kept_text = {}

    for _, document_pages in pages.groupby("document_id", sort=False):
        state = "before"
        for index, page_text in document_pages["page_text"].items():
            text = page_text.replace("Ɵ", "ti")

            if state == "before":
                start = BOARD_SECTION_START.search(text)
                if start is None:
                    kept_text[index] = ""
                    continue
                state = "inside"
                text = text[start.start():]

            if state == "inside":
                annex = ANNEX_START.search(text)
                if annex is not None:
                    state = "after"
                    text = text[:annex.start()]
                kept_text[index] = text.rstrip()
            else:
                kept_text[index] = ""

    pages["page_text"] = pd.Series(kept_text)
    pages["n_characters"] = pages["page_text"].str.len()
    return pages


# A board member's name on a line of its own, in capitals, opening their
# statement: "IRENE ESPINOSA CANTELLANO", or in 2018 "MTRA. IRENE ESPINOSA
# CANTELLANO". Checked on all 45 transcripts: five speakers each, except
# December 2018, when the board had four members.
SPEAKER_HEADING = re.compile(
    r"^[ \t]*(?:(?:LIC|MTRA|MTRO|DR|DRA)\.\s+)?"
    r"([A-ZÁÉÍÓÚÑ]+(?:[ \t]+[A-ZÁÉÍÓÚÑ]+){1,5})[ \t]*$",
    re.MULTILINE)


def keep_statements(pages):
    """Cut every transcript down to the board members' statements.

    Drops the header, the attendee list and the note on speaking order: all
    text before the first speaker's name. Same page-table-in, page-table-out
    shape as `keep_board_discussion`.
    """
    pages = pages.sort_values(["document_id", "page_number"]).copy()
    kept_text = {}

    for _, document_pages in pages.groupby("document_id", sort=False):
        started = False
        for index, text in document_pages["page_text"].items():
            if not started:
                first_speaker = SPEAKER_HEADING.search(text)
                if first_speaker is None:
                    kept_text[index] = ""
                    continue
                started = True
                text = text[first_speaker.start():]
            kept_text[index] = text

    pages["page_text"] = pd.Series(kept_text)
    pages["n_characters"] = pages["page_text"].str.len()
    return pages


def split_by_speaker(documents):
    """One row per (transcript, board member): who said it, and what.

    `documents` needs `document_id` and `document_text`. Speakers are numbered
    in the order they spoke, which by Banxico's rule is seniority among the
    deputy governors, then the governor.
    """
    records = []
    for document in documents.itertuples(index=False):
        headings = list(SPEAKER_HEADING.finditer(document.document_text))
        for order, heading in enumerate(headings, start=1):
            end = headings[order].start() if order < len(headings) else None
            statement = document.document_text[heading.end():end].strip()
            records.append(
                {
                    "document_id": document.document_id,
                    "speaker_order": order,
                    "speaker": heading.group(1).title(),
                    "statement_text": statement,
                    "n_words": len(statement.split()),
                }
            )
    return pd.DataFrame(records)


def parse_decision_date(title):
    """The decision date in a title like '... anunciada el 10 de agosto de 2023'."""
    if not title:
        return pd.NaT
    match = re.search(r"anunciada el (\d{1,2}) de (\w+) de (\d{4})", title.lower())
    if not match or match.group(2) not in SPANISH_MONTHS:
        return pd.NaT
    day, month, year = int(match.group(1)), SPANISH_MONTHS[match.group(2)], int(match.group(3))
    return pd.Timestamp(year=year, month=month, day=day)


def build_document_id(row):
    """The stable id for one document, e.g. banxico_min_2023_08_10.

    The positional fallback is for a row whose decision date could not be
    parsed; deliberately ugly, because it means the title format changed.
    """
    if pd.notna(row["decision_date"]):
        return f"{row['id_prefix']}_{row['decision_date']:%Y_%m_%d}"
    return f"{row['id_prefix']}_{row.name:03d}"


def scrape_index(page_url=PAGE_URL, headers=HEADERS):
    """Scrape the minutes page: one row per minuta and one per transcript.

    Filter on `document_type` to get either corpus on its own.
    """
    tree = fetch_page_html(page_url, headers=headers, encoding="latin-1")

    records = []
    # Each decision is one table row: date cell, then a cell with the title
    # followed by a "Minuta" link and, for 2018-2023, a "Transcripción" link.
    for row in tree.xpath("//tr[.//a[normalize-space(string(.))='Minuta']]"):
        cells = row.xpath("./td | ./th")
        if len(cells) < 2:
            continue

        date_text = clean_text(cells[0].text_content())
        date_match = re.search(r"\b\d{2}/\d{2}/\d{2}\b", date_text)

        # The title is the cell's own text, before the links.
        title = clean_text(cells[1].text)

        for link in cells[1].xpath(".//a[@href]"):
            link_text = clean_text(link.text_content())
            if link_text not in DOCUMENT_KINDS:
                continue
            document_type, id_prefix = DOCUMENT_KINDS[link_text]

            records.append(
                {
                    "publication_date_raw": date_match.group(0) if date_match else None,
                    "title": title,
                    "resource_type": link_text,
                    "pdf_url": urljoin(page_url, link.get("href")),
                    "source_page_url": page_url,
                    "institution": "Banco de México",
                    "document_type": document_type,
                    "id_prefix": id_prefix,
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
    metadata["publication_month"] = metadata["publication_date"].dt.to_period("M").astype("string")

    # The quarter a document belongs to is the quarter of the decision it discusses.
    metadata["decision_date"] = metadata["title"].apply(parse_decision_date)
    metadata["reference_year"] = metadata["decision_date"].dt.year
    metadata["quarter"] = metadata["decision_date"].dt.quarter

    metadata["document_variant"] = metadata["document_type"].map(
        {"Minuta": "minutes", "Transcripción": "transcript"})

    metadata["document_id"] = metadata.apply(build_document_id, axis=1)

    columns = [
        "document_id",
        "institution",
        "document_type",
        "document_variant",
        "title",
        "publication_date",
        "publication_month",
        "decision_date",
        "reference_year",
        "quarter",
        "resource_type",
        "pdf_url",
        "source_page_url",
    ]

    return metadata[columns]
