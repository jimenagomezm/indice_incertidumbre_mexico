"""Semantic similarity (embedding-based) index: Methodology 2.

Chunks and short conceptual "anchor" sentences are embedded with the same model;
a chunk's score is how much closer it sits to the positive (uncertainty) anchors
than to the negative (certainty) anchors.

Anchors are parameters, not hardcoded constants, so a different concept
(Hawkish/Dovish stance, ...) can be scored by writing new anchor sentences
without touching this module.

Aggregating chunk scores up to a document is deliberately NOT done here - see
index/common.py. That choice is a methodological question, not a property of the
embeddings (project_description.md section 4.3).
"""


# Monetary Policy Uncertainty Anchors 

# Anchors specific to Monetary Policy Uncertainty, used in parallel with the
# general-uncertainty anchors above (not replacing them). Unlike the general
# set, BOTH the positive and negative MPU anchors are about monetary policy:
# the co-occurrence relevance filter (src/exact_matching.py) already selects
# monetary-policy chunks, so the anchors' job is narrower — separate
# *uncertain* monetary-policy language from *confident/settled* monetary-policy
# language. Each positive anchor targets one dimension from
# project_description.md section 8 (future inflation, inflation expectations,
# the rate path, the balance of risks, growth, the exchange rate, external and
# financial conditions). Wording is kept in Banxico's formal register so the
# anchors sit close to the report text in embedding space.
MPU_POSITIVE_ANCHORS = [
    # future inflation
    "Existe incertidumbre sobre la trayectoria futura de la inflación y el momento en que convergerá a la meta.",
    # inflation expectations
    "Las expectativas de inflación podrían desanclarse, lo que complica la conducción de la política monetaria.",
    # rate path / policy decisions
    "No hay certeza sobre la trayectoria futura de la tasa de referencia ni sobre el momento de los próximos ajustes.",
    # balance of risks
    "El balance de riesgos para la inflación está sesgado al alza y su evolución es difícil de anticipar.",
    # growth / activity
    "La evolución de la actividad económica es incierta y podría alterar la postura de política monetaria.",
    # exchange rate
    "La volatilidad del tipo de cambio representa un riesgo para la inflación y para las decisiones de política monetaria.",
    # external / financial conditions
    "Choques externos y condiciones financieras globales más restrictivas podrían modificar el rumbo de la política monetaria.",
]

MPU_NEGATIVE_ANCHORS = [
    "La Junta de Gobierno decidió mantener la tasa de referencia, en línea con lo previsto y sin cambios en el panorama.",
    "La inflación evolucionó conforme a lo esperado y se prevé que converja a la meta de manera ordenada.",
    "Las expectativas de inflación permanecen ancladas y estables.",
    "El documento describe datos históricos y decisiones de política monetaria ya adoptadas, sin expresar dudas sobre el futuro.",
    "El panorama para la política monetaria es claro y la trayectoria esperada de la tasa se mantiene sin cambios.",
]


def embed_anchors(model, anchors):
    """Embed anchor sentences as e5-style queries (normalized, for cosine similarity)."""
    # e5 models expect anchors (the thing you search *for*) prefixed "query: ".
    inputs = [f"query: {text}" for text in anchors]
    # normalize_embeddings=True makes every vector length 1, so a later dot
    # product between two vectors equals their cosine similarity.
    return model.encode(inputs, normalize_embeddings=True, convert_to_numpy=True)


def embed_chunks(model, chunk_texts, batch_size=32, show_progress_bar=True):
    """Embed chunk texts as e5-style passages (normalized, for cosine similarity)."""
    # e5 models expect the documents being searched prefixed "passage: ".
    inputs = [f"passage: {text}" for text in chunk_texts]
    return model.encode(
        inputs,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def score_chunks(chunks, chunk_embeddings, positive_embeddings, negative_embeddings):
    """Add positive/negative/semantic_uncertainty_score columns to `chunks`.

    Because embeddings are normalized, the dot product is the cosine
    similarity. Each chunk's similarity to a group of anchors is its
    maximum similarity to any single anchor in that group. The semantic
    score is not a probability — only its relative order across chunks
    is meaningful.
    """
    chunks = chunks.copy()

    # Each entry is cosine similarity between one chunk and one anchor.
    # Shape: (n_chunks, n_positive_anchors) and (n_chunks, n_negative_anchors).
    positive_similarity_matrix = chunk_embeddings @ positive_embeddings.T
    negative_similarity_matrix = chunk_embeddings @ negative_embeddings.T

    # A chunk's similarity to a group = its closest single anchor in that group.
    chunks["positive_similarity"] = positive_similarity_matrix.max(axis=1)
    chunks["negative_similarity"] = negative_similarity_matrix.max(axis=1)
    # Net score: how much more "uncertainty-like" than "certainty-like" the
    # chunk reads. Only the ranking of these values across chunks is meaningful.
    chunks["semantic_uncertainty_score"] = (
        chunks["positive_similarity"] - chunks["negative_similarity"]
    )

    return chunks
