"""Methodology 1 - the dictionary method.

Counts uncertainty words that sit next to a word from one of five categories,
expressed as a rate per 1,000 words. Five term lists in, five series out.

    uncertainty + policy        ->  M1a, the general index
    uncertainty + inflation     ->  M1b, the headline topic
    uncertainty + cambiario
    uncertainty + growth
    uncertainty + geopolitical

**Why it is built this way.** Baker-Bloom-Davis (EPU) and Husted et al. (MPU)
both work on newspapers, where the hard problem is finding the ~2% of articles
that concern the central bank at all - which is what their "policy" and
"institution" conjuncts do. In a Banxico report that job is already done by the
choice of corpus: every page is the central bank writing about monetary policy,
so those conjuncts match nearly everything and discriminate nothing. Measured
directly, requiring the institution term made the index *worse*.

Cieslak, Hansen, McMahon & Xiao (NBER 31849) face the same situation on FOMC
transcripts and answer it by asking which TOPIC the uncertainty attaches to,
counting words rather than documents, within a sentence rather than anywhere in
a passage. This module follows that design, with one departure: a `policy`
category is kept alongside the topics so the EPU-style question ("uncertainty
about policy") is still asked, using identical machinery. It is one more term
list, not a second method.

**Two normalization choices worth knowing.**

Words per 1,000 words, not share of chunks. Banxico's reports have roughly
doubled in length since 2000, so a share-of-chunks denominator drifts upward for
reasons unrelated to uncertainty. A word rate does not. It also means this
method needs no chunking at all, and so escapes the aggregation problem that
methods 2 and 3 still have (project_description.md section 4.3).

Same sentence, or one either side. Inside a ~3,000-character passage almost any
two words co-occur, so proximity is what makes the attachment mean something.

**The term lists** were grown from a few seed words using word vectors trained on
this corpus (index/term_expansion.py) and then pruned by hand. Candidates and
decisions are recorded in data/terms/. No word appears in both the uncertainty
list and a category list - if one did, it would satisfy both halves of the
co-occurrence by itself.
"""

import re

import pandas as pd


# 58 terms, grown from 7 seeds. Includes the subjunctive hedging
# ("pudiera", "pudieran") that Spanish central-bank prose uses to express doubt
# far more often than the noun "incertidumbre".
DEFAULT_UNCERTAINTY_TERMS = [
        "acentuado", "acontecimientos", "adverso", "afectar", "aversión", "balance",
        "complejidad", "complejo", "complicado", "continúa", "deteriorado",
        "deterioro", "deterioró", "dificulta", "difícil", "duda", "elevada",
        "enfrenta", "enfrentado", "enfrentando", "episodios", "evento", "exposición",
        "incertidumbre", "incierta", "inciertas", "incierto", "inciertos",
        "incumplimiento", "inflacionario", "inflacionarios", "intensificación",
        "manifestado", "materialización", "panorama", "percepción", "persiste",
        "persisten", "perspectivas", "posibilidad", "posible", "preocupación",
        "presiones", "prevalece", "prevalecen", "prevista", "profundidad", "pudiera",
        "pudieran", "pérdidas", "repercusiones", "retos", "riesgo", "riesgos",
        "sesgado", "sesgo", "tensiones", "volatilidad"
]

# What the uncertainty must attach to. `policy` gives the general index; the
# other four are topics.
DEFAULT_COOCCURRENCE_CATEGORIES = {
    "policy": [
        "autoridad", "conducción", "convergencia", "decisiones", "decisión",
        "desinflacionario", "estrategia", "futuras", "junta", "mantener", "meta",
        "monetaria", "objetivo", "objetivos", "política", "postura", "referencia",
        "restricción", "restrictiva", "restrictivo", "reunión", "tasa"
    ],
    "inflation": [
        "agregada", "aumentos", "ausencia", "energéticos", "esperada", "expectativas",
        "general", "horizonte", "incrementos", "inflacionarias", "inflación", "meta",
        "perturbaciones", "precios", "primas", "subyacente", "subíndice",
        "trayectoria"
    ],
    "cambiario": [
        "apreciación", "argentino", "brasileño", "cambiario", "chileno", "chino",
        "colombiano", "coreano", "corona", "depreciación", "divisa", "divisas", "dxy",
        "dólar", "intervenciones", "lira", "mexicano", "moneda", "paridad", "peso",
        "polaco", "rand", "rublo", "rupia", "ruso", "tipo de cambio", "turca", "won",
        "yuan", "zloty"
    ],
    "growth": [
        "actividad", "agregada", "crecimiento", "debilidad", "debilitamiento",
        "demanda", "desaceleración", "economía", "económica", "empleo", "empleos",
        "expansión", "industrial", "laboral", "oferta", "producción", "reactivación",
        "recuperación", "trabajo"
    ],
    "geopolitical": [
        "agravamiento", "bélico", "conflicto", "conflictos", "disrupciones",
        "electoral", "escalamiento", "fragmentación", "geopolítica", "geopolíticas",
        "geopolítico", "geopolíticos", "guerra", "impuestas", "inestabilidad",
        "interrupciones", "irak", "irán", "militar", "oriente", "políticos",
        "recrudecimiento", "sanciones", "surgimiento", "ucrania"
    ],
}


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def build_term_pattern(terms):
    """Combine a term list into one case-insensitive alternation.

    `re.escape` protects any regex-special characters, and the word boundary
    stops "riesgo" matching inside "riesgoso". Multi-word terms such as
    "tipo de cambio" work unchanged, which matters: the single words `cambio`,
    `tipo` and `real` catch mostly non-exchange-rate text ("sin cambio",
    "PIB real"), while the phrase does not.
    """
    return r"\b(?:" + "|".join(re.escape(term) for term in terms) + r")"


def split_into_sentences(text):
    """Split text into sentences. Deliberately crude, and that is fine.

    The measure only needs to know whether two words are close together, so a
    boundary landing one sentence early or late changes little - especially as
    the window below already spans neighbouring sentences. A proper Spanish
    sentence tokenizer would add a dependency for no measurable gain.
    """
    return [s.strip() for s in SENTENCE_BOUNDARY.split(text) if s.strip()]


def count_uncertainty_by_category(text, uncertainty_terms=None,
                                  categories=None, window=1):
    """Count uncertainty words, attributed to the category they sit next to.

    An uncertainty word counts for category k if a category-k word appears in
    the same sentence, or within `window` sentences either side. The window
    exists because a thought often runs across a sentence break: "La inflación
    se ubicó por encima de lo previsto. Este resultado es incierto." - the doubt
    belongs to inflation even though the two words are in different sentences.

    A word can count for more than one category (a sentence may mention both
    inflation and the exchange rate), so the counts deliberately do not sum to
    the total. There is no blended "any category" figure: with five lists,
    "next to any of them" is almost no constraint, and the resulting series
    collapses into plain word-counting. PMU reports its topics separately for
    the same reason.

    Returns a count per category, plus `total_words`.
    """
    if uncertainty_terms is None:
        uncertainty_terms = DEFAULT_UNCERTAINTY_TERMS
    if categories is None:
        categories = DEFAULT_COOCCURRENCE_CATEGORIES

    sentences = split_into_sentences(text.lower())
    uncertainty_pattern = re.compile(build_term_pattern(uncertainty_terms))
    category_patterns = {name: re.compile(build_term_pattern(terms))
                         for name, terms in categories.items()}

    # Which categories each sentence mentions - computed once per sentence
    # rather than once per uncertainty word.
    categories_in_sentence = [
        {name for name, pattern in category_patterns.items() if pattern.search(sentence)}
        for sentence in sentences
    ]

    counts = {name: 0 for name in categories}
    for position, sentence in enumerate(sentences):
        n_uncertainty = len(uncertainty_pattern.findall(sentence))
        if not n_uncertainty:
            continue
        start, end = max(0, position - window), min(len(sentences), position + window + 1)
        for name in set().union(*categories_in_sentence[start:end]):
            counts[name] += n_uncertainty

    counts["total_words"] = len(text.split())
    return counts


def compute_cooccurrence_rates(documents, text_column="document_text",
                               uncertainty_terms=None, categories=None,
                               window=1, per_words=1000):
    """Methodology 1 at the document level: one rate per category.

    Returns one row per document with `unc_rate_<category>` for each category,
    plus the word count so a reader can see how much text each rate rests on.
    """
    if categories is None:
        categories = DEFAULT_COOCCURRENCE_CATEGORIES

    records = []
    for row in documents.itertuples(index=False):
        counts = count_uncertainty_by_category(
            getattr(row, text_column), uncertainty_terms, categories, window)
        total_words = max(counts.pop("total_words"), 1)
        record = {"document_id": row.document_id, "n_words": total_words}
        for name, count in counts.items():
            record[f"unc_rate_{name}"] = count / total_words * per_words
        records.append(record)

    return pd.DataFrame(records)
