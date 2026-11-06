"""Steps shared by more than one methodology.

Small on purpose. Each method now owns its own aggregation, so what is left here
is genuinely common: reassembling parent chunks, the LLM aggregation, and the
standardization every index ends with.

On the *aggregation problem* (project_description.md section 4.3) - turning
chunk-level evidence into one number per document - the three methods answer it
differently, and each answer lives with its method:

- **M1** never faces it. It counts words and divides by words, so there are no
  chunk scores to combine.
- **M2** takes the mean of its own top-decile chunks: the intensity of a
  report's most uncertain passages.
- **M3** averages over the chunks the model itself marked relevant.

That independence is deliberate. Any one of them can be run on a new corpus
without the other two.
"""

import pandas as pd

DOCUMENT_GROUP_COLUMNS = ["document_id", "publication_date", "title"]


def rebuild_parent_chunks(chunks, text_column="chunk_text"):
    """Reassemble the ~3,000-character parent chunks from their sub-chunks.

    Sub-chunks exist only because the embedding model has a hard token limit.
    They are the wrong unit for dictionary matching: a triple whose three terms
    land either side of a sub-chunk boundary would simply be missed. So
    co-occurrence is measured on the parent, rebuilt here by re-joining its
    pieces in order.

    The pieces are joined with a single space, which is how they were scored
    before, so the reconstructed text matches what the LLM was given.
    """
    return (
        chunks.groupby("parent_chunk_id")
        .agg(
            document_id=("document_id", "first"),
            publication_date=("publication_date", "first"),
            title=("title", "first"),
            document_variant=("document_variant", "first"),
            **{text_column: (text_column, lambda pieces: " ".join(pieces))},
        )
        .reset_index()
    )


def top_decile_mean(values, decile=0.90):
    """Mean of the top 10% of a document's chunk scores.

    Ties at the cut-off are included (>= rather than >), so a document whose
    scores bunch together keeps all of the tied chunks rather than an arbitrary
    subset of them.
    """
    cutoff = values.quantile(decile)
    return values.loc[values >= cutoff].mean()


def aggregate_by_top_decile(scored_chunks, score_column, output_column,
                            groupby_cols=None, decile=0.90):
    """Aggregate chunk scores to one number per document, using the top decile.

    This is the embedding method's answer to the aggregation problem, and it is
    deliberately self-contained: the method's own scores choose the method's own
    chunks, so it can run on a new corpus without any other method having been
    run first.

    Why not a plain mean. Most of a quarterly report is tables, historical data
    and methodology. Those chunks are not *certain* - they are not about the
    outlook at all - but they score near zero, so averaging them in treats "does
    not discuss the outlook" as "is confident about the outlook". That is simply
    wrong, and with ~80% of chunks contributing near zero it swamps the few that
    carry signal.

    What this measures, stated precisely: the INTENSITY of a report's most
    uncertain passages, not how widespread uncertainty is across the report.
    Two reports can differ because one is mildly uncertain throughout and the
    other is intensely uncertain in a few places; this ranks the second higher.
    The plain mean is reported alongside so that choice stays visible.

    The assumption doing the work is that a document's highest-scoring chunks
    are the ones genuinely discussing the outlook. That is untested. A relevance
    gate built from its own anchor sentences would measure it directly and is
    the natural thing to revisit once there are more documents per quarter.
    """
    if groupby_cols is None:
        groupby_cols = DOCUMENT_GROUP_COLUMNS

    return scored_chunks.groupby(groupby_cols, as_index=False).agg(
        **{output_column: (score_column, lambda v: top_decile_mean(v, decile)),
           f"{output_column}_plain_mean": (score_column, "mean"),
           "n_chunks": (score_column, "size")}
    )


def aggregate_llm_scores(llm_chunk_scores, output_column="mpu_llm"):
    """Methodology 3 at the document level: mean uncertainty score over the
    chunks the model judged relevant.

    Two filters, for different reasons. `relevant` is the model's own judgment
    that the chunk is about the macro / monetary-policy outlook - the rubric
    tells it to leave the score empty otherwise. `uncertainty_score.notna()`
    additionally drops rows where the call failed and no verdict came back;
    those are stored with a null `relevant` so that failures stay visible in the
    data instead of quietly vanishing.

    Note `relevant` is an object column holding True/False/None, so it is
    compared to True explicitly rather than used as a boolean mask.
    """
    scored_and_relevant = llm_chunk_scores[
        (llm_chunk_scores["relevant"] == True)  # noqa: E712 - object column
        & (llm_chunk_scores["uncertainty_score"].notna())
    ]
    return scored_and_relevant.groupby("document_id", as_index=False).agg(
        **{output_column: ("uncertainty_score", "mean")}
    )


def standardize_to_index(values, index_mean=100, index_scale=10):
    """Convert a raw score series into an index centered at `index_mean`, where
    one standard deviation of the historical series equals `index_scale` points.

    This is the EPU presentation convention, and it is what makes series built
    from different methods comparable on one chart: each is expressed in its own
    standard deviations rather than in its own raw units.

    `values.std()` is pandas' sample standard deviation (ddof=1). Switching to
    numpy's population default (ddof=0) would rescale every point by
    sqrt(n/(n-1)) - a silent change to every published number.
    """
    # Always compute in float64: dividing by a small standard deviation
    # amplifies any precision lost upstream, and some score columns arrive as
    # float32 because they come from embedding vectors.
    values = values.astype("float64")
    z_scores = (values - values.mean()) / values.std()
    return index_mean + index_scale * z_scores
