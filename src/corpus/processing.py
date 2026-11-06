"""Extract and clean text from downloaded Banxico report PDFs."""

import re
from pathlib import Path

import fitz
import pandas as pd
from tqdm.auto import tqdm


def clean_pdf_text(text):
    """Clean raw text extracted from one PDF page, preserving paragraphs."""
    if not text:
        return ""

    # remove soft hyphens (invisible line-break hyphenation, e.g. "incerti-")
    text = text.replace("­", "")

    # rejoin words split by hyphenation across a line break
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # normalize repeated spaces from justified text, keep paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# Blocks whose text starts with one of these are figure/table captions or
# source/note lines, not body prose — dropped when drop_noise=True.
# Spanish caption vocabulary. A parameter rather than a fixed constant because
# a different source has a different noise profile - meeting minutes, for
# instance, are nearly all prose and have very few figures.
DEFAULT_FIGURE_MARKERS = re.compile(
    r"^\s*(gr[aá]fica|cuadro|tabla|figura|fuente|nota)\b", re.IGNORECASE)


def _letter_ratio(text):
    """Share of non-space characters that are letters (low = numeric/label noise)."""
    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return 0.0
    return sum(c.isalpha() for c in non_space) / len(non_space)


def _is_noise_block(text, y0, y1, page_height, header_frac=0.08, footer_frac=0.08,
                    noise_markers=DEFAULT_FIGURE_MARKERS):
    """Heuristically flag header/footer, figure captions, and axis-label blocks.

    Conservative on purpose: only drops blocks that are clearly not body prose
    (page header/footer bands, "Gráfica/Cuadro/Fuente" lines, and short or
    mostly-numeric blocks like chart axis ticks and year rows).
    """
    stripped = text.strip()
    if not stripped:
        return True
    # repeating header/footer bands (page number, institution name)
    if y1 < page_height * header_frac or y0 > page_height * (1 - footer_frac):
        return True
    # figure/table caption or source/note lines
    if noise_markers.match(stripped):
        return True
    ratio = _letter_ratio(stripped)
    # short mostly-numeric blocks (axis ticks) or long number-only rows (year axes)
    if (len(stripped) < 25 and ratio < 0.4) or ratio < 0.2:
        return True
    return False


def extract_page_columns(page, gutter_ratio=0.5, min_col_blocks=2, drop_noise=True,
                         header_frac=0.08, footer_frac=0.08,
                         noise_markers=DEFAULT_FIGURE_MARKERS):
    """Extract a page's text in human reading order, handling two columns.

    Banxico reports are largely two-column. `get_text("text", sort=True)`
    reads horizontally across both columns and interleaves them, shredding
    sentences. Instead we take the layout BLOCKS (which PyMuPDF already
    separates by column) and read the LEFT column top-to-bottom, then the
    RIGHT column. Single-column pages are read straight top-to-bottom.

    With `drop_noise=True`, header/footer/figure/axis-label blocks are
    filtered out first (see `_is_noise_block`); set it False to keep raw
    blocks (useful for comparing 2a-only vs. 2b).

    Each block is `(x0, y0, x1, y1, text, block_no, block_type)`.
    """
    page_height = page.rect.height
    blocks = [b for b in page.get_text("blocks") if b[4].strip()]
    if drop_noise:
        blocks = [b for b in blocks
                  if not _is_noise_block(b[4], b[1], b[3], page_height,
                                         header_frac, footer_frac, noise_markers)]
    if not blocks:
        return ""

    midline = page.rect.width * gutter_ratio
    left_blocks = [b for b in blocks if (b[0] + b[2]) / 2 < midline]
    right_blocks = [b for b in blocks if (b[0] + b[2]) / 2 >= midline]

    # treat as two-column only if BOTH sides carry real text; otherwise read
    # the whole page as a single column (top-to-bottom, then left-to-right)
    is_two_column = len(left_blocks) >= min_col_blocks and len(right_blocks) >= min_col_blocks
    if is_two_column:
        ordered = sorted(left_blocks, key=lambda b: b[1]) + sorted(right_blocks, key=lambda b: b[1])
    else:
        ordered = sorted(blocks, key=lambda b: (b[1], b[0]))

    # join blocks as paragraphs so the chunker's blank-line splitting still works
    return "\n\n".join(b[4].strip() for b in ordered)


# Document-level fields copied onto every page row, so a page is
# self-describing. A parameter rather than a hardcoded list: this is the exact
# thing that used to make the extractor quarterly-only, because it read
# `row.reference_year` and `row.quarter` directly and any source without those
# columns raised AttributeError.
DEFAULT_PAGE_METADATA_COLUMNS = [
    "document_id",
    "institution",
    "document_type",
    "title",
    "publication_date",
    "publication_month",
    "reference_year",
    "quarter",
    "local_pdf_path",
    "pdf_url",
]


def extract_pdf_pages(metadata, carry_columns=DEFAULT_PAGE_METADATA_COLUMNS,
                      extract_page_text=extract_page_columns, column_options=None):
    """Extract text from every page of every downloaded PDF in `metadata`.

    Returns one row per page with the document's metadata attached.

    `carry_columns` says which metadata fields to copy onto each page.
    `extract_page_text` and `column_options` let a caller change how a page is
    read - for example `column_options={"drop_noise": False}` to keep figure
    captions and axis labels, which is how the noise filter was compared
    against raw extraction in the first place. Previously those settings were
    unreachable, because the extractor called `extract_page_columns(page)` with
    no arguments.
    """
    if column_options is None:
        column_options = {}

    page_records = []

    # Only process rows whose PDF was actually downloaded.
    valid_metadata = metadata[metadata["local_pdf_path"].notna()]

    # Iterate as dicts rather than namedtuples so the carried fields can be
    # looked up by name from a list, instead of being written out one by one.
    for row in tqdm(valid_metadata.to_dict("records"), total=len(valid_metadata)):
        pdf_path = Path(row["local_pdf_path"]) if row["local_pdf_path"] else None
        if not pdf_path or not pdf_path.exists():
            # Path recorded but the file is missing on disk; skip it.
            continue

        document_fields = {column: row[column] for column in carry_columns}
        document_fields["local_pdf_path"] = str(pdf_path)

        try:
            document = fitz.open(pdf_path)

            # enumerate(..., start=1) gives 1-based page numbers, matching how
            # a reader would refer to a page.
            for page_number, page in enumerate(document, start=1):
                raw_text = extract_page_text(page, **column_options)
                page_text = clean_pdf_text(raw_text)

                page_records.append({
                    **document_fields,
                    "page_number": page_number,
                    "page_text": page_text,
                    "n_characters": len(page_text),
                })

            document.close()

        except Exception as error:
            # One bad PDF should not kill the whole run: log it and move on.
            print(f"Error extracting {row['document_id']}: {error}")

    return pd.DataFrame(page_records)


def flag_page_quality(pages):
    """Add quality-control flags used to decide which pages feed into chunking."""
    pages = pages.copy()
    # Almost no text: likely a blank or cover page.
    pages["is_near_empty"] = pages["n_characters"] < 50
    # Thin page (e.g. a section divider); flagged but not necessarily dropped.
    pages["is_short_page"] = pages["n_characters"] < 200
    # The replacement character "�" signals text was garbled during extraction.
    pages["has_encoding_error"] = pages["page_text"].str.contains("�", na=False)
    # A page is trusted only if it has real text AND no encoding error.
    pages["is_usable_page"] = pages["page_text"].str.strip().ne("") & ~pages["has_encoding_error"]
    return pages
