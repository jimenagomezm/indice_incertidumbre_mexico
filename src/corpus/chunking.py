"""Split long documents into chunks small enough for embeddings and LLM scoring.

Two stages, and they exist for different reasons:

1. `create_document_chunks` builds **parent chunks** of roughly 3,000 characters
   by filling them with whole paragraphs. This is the unit a human would
   recognise as "a passage" - big enough to carry an argument, small enough to
   score. It is the unit the LLM reads and the unit dictionary matching uses.

2. `split_chunks_by_tokens` cuts those parents into **sub-chunks** that fit the
   embedding model's hard token limit (512 for multilingual-e5-base). This is
   purely a technical constraint of one model, which is why every sub-chunk
   keeps a `parent_chunk_id`: the other two methods reassemble the parent and
   ignore the split entirely.

Chunks are an intermediate unit, never the final one. Turning chunk scores into
one number per document is a separate, methodological question - see
index/common.py and project_description.md section 4.3.

A warning worth reading before editing anything here. The parent `chunk_id` is
the join key to the LLM scores in
`data/processed/banxico/mpu_llm_chunk_scores.parquet`, which cost real money to
produce. Change how chunk boundaries fall and the ids shift, the join silently
produces empty values instead of failing, and every published index number moves
with no error raised. `scripts/check_refactor.py` fingerprints all 7,809 chunk
texts precisely to catch that.
"""

import re

import pandas as pd
from tqdm.auto import tqdm

# Document-level fields copied onto every chunk, so a chunk row is
# self-describing and downstream code never needs a second join back to the
# metadata. It is a parameter rather than a hardcoded list because a corpus that
# is not quarterly (meeting minutes, say) carries different fields.
DEFAULT_CHUNK_METADATA_COLUMNS = [
    "institution",
    "document_type",
    "document_variant",
    "title",
    "publication_date",
    "publication_month",
    "reference_period",
    "reference_year",
    "quarter",
    "pdf_url",
]


def split_long_text(text, max_chars=3000):
    """Split a single paragraph that is by itself longer than max_chars.

    Rare - most paragraphs are far shorter - but a stray one would otherwise
    blow past the size limit on its own. Cuts at a blank space so words stay
    intact.
    """
    text = text.strip()
    if not text:
        return []

    pieces = []
    while len(text) > max_chars:
        # Prefer the last blank space before the limit, so we don't slice
        # through the middle of a word.
        split_position = text.rfind(" ", 0, max_chars)

        # If the nearest space is very far back - in the first 60% of the
        # window - there is no good break point (a long unbroken string like a
        # table of numbers), so cut hard at the limit instead of throwing away
        # most of the window. The 0.60 is a judgement call; changing it changes
        # how such passages are cut.
        if split_position < max_chars * 0.60:
            split_position = max_chars

        piece = text[:split_position].strip()
        if piece:
            pieces.append(piece)
        text = text[split_position:].strip()

    if text:
        pieces.append(text)

    return pieces


def split_page_into_paragraphs(page_text):
    """Split one page into paragraphs on blank lines.

    This is why `clean_pdf_text` is careful to preserve blank lines: they are
    the only paragraph signal a PDF gives us.
    """
    return [paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", page_text)
            if paragraph.strip()]


def iter_document_pieces(document_pages, max_chars):
    """Walk one document and yield `(piece_of_text, page_number)` in reading order.

    The document is naturally nested - pages contain paragraphs, and the odd
    paragraph has to be split again - so flattening it costs three short loops.
    Doing that flattening *here*, in a generator whose only job is flattening,
    is what lets the function that actually builds chunks be a single flat loop
    over one stream of pieces.
    """
    for page in document_pages.itertuples(index=False):
        page_text = page.page_text.strip()
        if not page_text:
            continue
        for paragraph in split_page_into_paragraphs(page_text):
            for piece in split_long_text(paragraph, max_chars=max_chars):
                yield piece, page.page_number


def make_chunk_record(document_id, chunk_number, parts, page_numbers):
    """Build one chunk row from the pieces collected for it.

    The only place a chunk record is constructed, so the id format and the
    field list are defined once.
    """
    chunk_text = "\n\n".join(parts).strip()
    return {
        "chunk_id": f"{document_id}_c{chunk_number:04d}",
        "document_id": document_id,
        "chunk_number": chunk_number,
        # A chunk can span a page break, so it records the range it came from
        # rather than a single page.
        "page_start": min(page_numbers),
        "page_end": max(page_numbers),
        "chunk_text": chunk_text,
        "n_characters": len(chunk_text),
    }


def accumulate_pieces_into_chunks(document_id, pieces, max_chars):
    """Fill chunks with consecutive pieces, closing one when the next would
    overflow.

    Deliberately greedy and order-preserving: pieces are never reordered or
    dropped, so a chunk is always a contiguous stretch of the report.
    """
    document_chunks = []
    current_parts = []   # pieces accumulating into the chunk being built
    current_pages = []   # the page each of those pieces came from

    for piece, page_number in pieces:
        # Measure what the chunk would actually look like with this piece added.
        # The "\n\n" separators are part of the stored text, so they have to be
        # part of the length test - summing the piece lengths would undercount.
        candidate_text = "\n\n".join(current_parts + [piece])

        if current_parts and len(candidate_text) > max_chars:
            document_chunks.append(make_chunk_record(
                document_id, len(document_chunks) + 1, current_parts, current_pages))
            # The overflowing piece becomes the seed of the next chunk.
            current_parts, current_pages = [piece], [page_number]
        else:
            current_parts.append(piece)
            current_pages.append(page_number)

    # Whatever is still buffered when the document ends is its final chunk.
    if current_parts:
        document_chunks.append(make_chunk_record(
            document_id, len(document_chunks) + 1, current_parts, current_pages))

    return document_chunks


def merge_short_final_chunk(document_chunks, min_chars):
    """Fold a stubby last chunk into the one before it.

    Documents rarely end on an exact chunk boundary, so the last chunk is often
    a fragment. Merging keeps it - and the pages it came from - traceable,
    rather than dropping text or leaving a chunk too short to score sensibly.

    Chunk numbers are deliberately *not* renumbered afterwards: the ids are the
    join key to the paid LLM scores, so they must stay stable.
    """
    if len(document_chunks) < 2:
        return document_chunks
    if document_chunks[-1]["n_characters"] >= min_chars:
        return document_chunks

    last_chunk = document_chunks[-1]
    merged_chunk = dict(document_chunks[-2])
    merged_chunk["chunk_text"] = (
        merged_chunk["chunk_text"] + "\n\n" + last_chunk["chunk_text"])
    merged_chunk["page_end"] = last_chunk["page_end"]
    merged_chunk["n_characters"] = len(merged_chunk["chunk_text"])

    return document_chunks[:-2] + [merged_chunk]


def attach_document_metadata(document_chunks, metadata_row, metadata_columns):
    """Copy the document's metadata onto each of its chunks."""
    for chunk in document_chunks:
        chunk.update({column: metadata_row[column] for column in metadata_columns})
    return document_chunks


def create_document_chunks(pages_df, metadata_df, max_chars=3000, min_chars=500,
                           metadata_columns=DEFAULT_CHUNK_METADATA_COLUMNS):
    """Turn page-level text into ~max_chars chunks, one document at a time.

    Each document is handled on its own so a chunk can never span two reports.
    The four steps below are each a named function above; this one only decides
    the order they run in.
    """
    metadata_lookup = metadata_df.set_index("document_id")
    all_chunk_records = []

    # sort=False keeps documents in the order they appear in pages_df.
    grouped_documents = pages_df.groupby("document_id", sort=False)

    for document_id, document_pages in tqdm(
            grouped_documents, total=pages_df["document_id"].nunique()):
        document_pages = document_pages.sort_values("page_number")

        pieces = iter_document_pieces(document_pages, max_chars)
        document_chunks = accumulate_pieces_into_chunks(document_id, pieces, max_chars)
        document_chunks = merge_short_final_chunk(document_chunks, min_chars)
        document_chunks = attach_document_metadata(
            document_chunks, metadata_lookup.loc[document_id], metadata_columns)

        all_chunk_records.extend(document_chunks)

    return pd.DataFrame(all_chunk_records)


def split_chunks_by_tokens(chunks, tokenizer, max_tokens=450):
    """Cut each parent chunk into sub-chunks of at most max_tokens tokens.

    Needed because the embedding model has a hard token limit (512 for
    multilingual-e5-base) that a chunk sized by *characters* can still exceed -
    Spanish text does not convert to tokens at a fixed rate.

    Every sub-chunk keeps `parent_chunk_id`, which is what lets the dictionary
    and LLM methods work on whole parents and ignore this split.
    """
    new_chunk_records = []

    for chunk in tqdm(chunks.itertuples(index=False), total=len(chunks)):
        # Ask the model's own tokenizer how this text breaks into tokens. No
        # special or padding tokens - we only want the real content.
        token_ids = tokenizer(
            chunk.chunk_text,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )["input_ids"]

        token_windows = [token_ids[start:start + max_tokens]
                         for start in range(0, len(token_ids), max_tokens)]

        for subchunk_number, window_ids in enumerate(token_windows, start=1):
            piece_text = tokenizer.decode(window_ids, skip_special_tokens=True).strip()
            if not piece_text:
                continue

            # Keep the parent's fields, then override the identity ones so each
            # sub-chunk stays traceable to the chunk it was split from.
            record = chunk._asdict()
            record["parent_chunk_id"] = chunk.chunk_id
            record["chunk_id"] = f"{chunk.chunk_id}_s{subchunk_number:02d}"
            record["subchunk_number"] = subchunk_number
            record["chunk_text"] = piece_text
            record["n_characters"] = len(piece_text)

            new_chunk_records.append(record)

    return pd.DataFrame(new_chunk_records)
