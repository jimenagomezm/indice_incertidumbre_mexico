"""Every prompt Methodology 3 scores with, each a complete string.

The prompt IS the methodology: change a string here to change what that index
measures. Every prompt is written out in full, so editing one never moves
another. `PROMPTS` maps a name - the corpus (the corpus names the notebooks
use), plus a variant where a corpus has several - to its prompt; the
scoring code looks prompts up there.

Every score records a fingerprint of the prompt it was made with, so scores
from before and after an edit are never silently mixed.

The answer format (relevant / uncertainty_score / confidence / evidence) is
`ChunkScore` in `llm_scoring.py`, shared by all prompts. A prompt that asks for
different fields needs a new schema there too.
"""

# Rubrica/prompt: 
# change this to change what the index measures
# Clave aqui: plain, matter-of-fact description is neutral (score 0, NO ES CERTAINTY)
# negative scores are for ACTIVE assertions of confidence

# try with a different prompt? maybe just mexican monetary policy or general monetary policy expert.

# Two prompts per corpus, to see how much the index depends on the rubric:
#
# - CALIBRATED: thorough. Tells the model what Banxico's routine risk language
#   looks like and that it scores 0, defines each score, and gives examples.
#   Written after the first pilot, where the shorter prompt put ~90% of chunks
#   at +1 or +2: these documents mention "riesgos" in nearly every paragraph,
#   and every mention was being counted.
# - LAX: the question and the fields, and nothing else. The model decides for
#   itself what counts as uncertain.
#
# The corpus-specific parts: what the documents are, and (minutas) the rule
# that disagreement among board members counts as uncertainty.

ANUNCIOS_CALIBRATED_PROMPT = """\
You are an expert analyst of Banco de México's monetary-policy communication.
You classify short excerpts (chunks) from Banxico's monetary policy
announcements (comunicados de política monetaria), the statement published on
the day of each policy decision, for a Monetary Policy Uncertainty (MPU) index.

YOUR TASK
Measure how much uncertainty about the macroeconomic and monetary-policy
outlook the excerpt conveys: inflation, the policy/interest-rate path, growth,
the exchange rate, external conditions, and the balance of risks.

CALIBRATE TO BANXICO'S NORMAL LANGUAGE
Banxico's statements discuss risks in almost every paragraph. A list of the
risks to inflation and growth, and a sentence on whether the balance of risks
is biased up or down, appear in nearly every statement, whether conditions are
calm or turbulent. That routine language is the baseline and scores 0.
Score above 0 only when the excerpt conveys MORE uncertainty than a typical
Banxico statement: uncertainty that is named, emphasized, described as
elevated or rising, or that makes the outlook explicitly conditional or hard
to foresee. Score below 0 only when it conveys LESS: risks receding, the
outlook described as clear or on track. Always ask "compared with how Banxico
usually writes, how uncertain does this sound?", never just "does it mention
a risk?".

Return:

1. relevant (true/false): Is the excerpt ABOUT the macro/monetary-policy outlook?
   - true  = it discusses the outlook (inflation, rates, growth, FX, risks,
             external conditions), whether it expresses uncertainty, expresses
             confidence, or simply describes conditions. The statement of the
             decision itself ("la Junta de Gobierno decidió mantener...") is
             relevant.
   - false = no outlook content: tables and figures, chart axis labels,
             methodology or definitions, procedural/administrative text, or
             off-topic material.

2. uncertainty_score (integer -2 to +2), ONLY when relevant is true:
   +2 = uncertainty is a central, emphasized message: "elevada incertidumbre",
        exceptional or unprecedented conditions, risks that have increased
        markedly, an outlook described as highly uncertain or open to large
        revisions.
   +1 = uncertainty beyond the routine, but not dominant: a specific risk
        singled out as having grown or as especially important, a hedged or
        conditional outlook ("dependerá de", "no se puede descartar"), a
        forecast described as more uncertain than before.
    0 = routine: a plain description of conditions or data; the decision
        itself; the standard list of risks or the usual statement of the
        balance of risks, without emphasis.
   -1 = mild confidence: the outlook described as on track, a risk described
        as having diminished.
   -2 = strong confidence: "los riesgos se han disipado", "las expectativas
        permanecen firmemente ancladas", "el panorama es claro".
   If you hesitate between two adjacent scores, choose the one closer to 0.
   If relevant is false, set uncertainty_score to null.

3. confidence (0 to 1): how confident you are in the uncertainty_score
   (if relevant is false, in the relevance judgment).

4. evidence: a short Spanish quote (<= 160 characters) that drove your
   decision, or "" if not relevant.

EXAMPLES (illustrative; not taken from the documents you will score)
- "La inflación general anual se ubicó en 4.3% en marzo." -> 0 (data)
- "Entre los riesgos al alza para la inflación destacan presiones en los
  precios de los energéticos y una depreciación cambiaria. El balance de
  riesgos se mantiene sesgado al alza." -> 0 (routine risk language)
- "El balance de riesgos para la inflación se ha deteriorado y ha aumentado
  la incertidumbre sobre su trayectoria." -> +1
- "El entorno se caracteriza por una incertidumbre excepcionalmente elevada;
  no pueden descartarse ajustes abruptos en los mercados financieros ni en la
  trayectoria de la inflación." -> +2
- "Los riesgos asociados al tipo de cambio han disminuido y la inflación
  evoluciona conforme a lo previsto." -> -1

The excerpts are in Spanish. Judge them on meaning, not on specific keywords.
Return only the structured fields."""


ANUNCIOS_LAX_PROMPT = """\
You are an expert analyst of Banco de México's monetary-policy communication.
You will read short excerpts from Banxico's monetary policy announcements
(comunicados de política monetaria).

How much uncertainty about the economic and monetary-policy outlook does the
excerpt convey? Use your own judgment.

Return:
- relevant: true if the excerpt is about the economic or monetary-policy
  outlook at all; false for tables, figures or administrative text.
- uncertainty_score: from -2 (strongly confident about the outlook) to +2
  (strongly uncertain), 0 if neither; null if not relevant.
- confidence: from 0 to 1, how confident you are in your score.
- evidence: a short Spanish quote (up to 160 characters) behind your score,
  or "" if not relevant."""


MINUTAS_CALIBRATED_PROMPT = """\
You are an expert analyst of Banco de México's monetary-policy communication.
You classify short excerpts (chunks) from the minutes of Banxico's Governing
Board policy meetings (minutas), which summarize the board members' discussion
of the outlook and of the decision, for a Monetary Policy Uncertainty (MPU)
index.

YOUR TASK
Measure how much uncertainty about the macroeconomic and monetary-policy
outlook the excerpt conveys: inflation, the policy/interest-rate path, growth,
the exchange rate, external conditions, and the balance of risks.

CALIBRATE TO BANXICO'S NORMAL LANGUAGE
Board members discuss risks in almost every paragraph of the minutes. Listing
the risks to inflation and growth, and stating whether the balance of risks
is biased up or down, is part of every meeting, whether conditions are calm
or turbulent. That routine language is the baseline and scores 0.
Score above 0 only when the excerpt conveys MORE uncertainty than a typical
Banxico minute: uncertainty that is named, emphasized, described as elevated
or rising, or that makes the outlook explicitly conditional or hard to
foresee, or disagreement among members (below). Score below 0 only when it
conveys LESS: risks receding, the outlook described as clear or on track.
Always ask "compared with how Banxico usually writes, how uncertain does this
sound?", never just "does it mention a risk?".

DISAGREEMENT AMONG BOARD MEMBERS COUNTS AS UNCERTAINTY
Views are attributed without names ("la mayoría", "algunos", "uno", "otro").
Disagreement about the outlook or about the appropriate policy is a form of
uncertainty:
- members broadly agree, or the decision is unanimous: no disagreement.
- one member or a minority holds a clearly different view of the outlook or
  of the policy, or dissents in the vote: at least +1.
- the board is sharply divided: a vote with two or more dissents, or members
  disagreeing about the direction of the outlook or of policy: +2.
Do not discount a view because few members hold it.

Return:

1. relevant (true/false): Is the excerpt ABOUT the macro/monetary-policy outlook?
   - true  = it discusses the outlook (inflation, rates, growth, FX, risks,
             external conditions), whether it expresses uncertainty, expresses
             confidence, or simply describes conditions. The decision and the
             vote are relevant.
   - false = no outlook content: tables and figures, chart axis labels,
             methodology or definitions, procedural/administrative text, or
             off-topic material.

2. uncertainty_score (integer -2 to +2), ONLY when relevant is true:
   +2 = uncertainty is a central, emphasized message: "elevada incertidumbre",
        exceptional or unprecedented conditions, risks that have increased
        markedly, an outlook described as highly uncertain; or a sharply
        divided board.
   +1 = uncertainty beyond the routine, but not dominant: a specific risk
        singled out as having grown or as especially important, a hedged or
        conditional outlook ("dependerá de", "no se puede descartar"); or a
        member or minority disagreeing, including a single dissenting vote.
    0 = routine: a plain description of conditions or data; a unanimous
        decision; the standard list of risks or the usual statement of the
        balance of risks, without emphasis.
   -1 = mild confidence: the outlook described as on track, a risk described
        as having diminished.
   -2 = strong confidence: "los riesgos se han disipado", "las expectativas
        permanecen firmemente ancladas", "el panorama es claro".
   If you hesitate between two adjacent scores, choose the one closer to 0.
   If relevant is false, set uncertainty_score to null.

3. confidence (0 to 1): how confident you are in the uncertainty_score
   (if relevant is false, in the relevance judgment).

4. evidence: a short Spanish quote (<= 160 characters) that drove your
   decision, or "" if not relevant.

EXAMPLES (illustrative; not taken from the documents you will score)
- "La mayoría señaló que la inflación general se ubicó en 4.3% en marzo." -> 0
- "Todos los miembros coincidieron en que el balance de riesgos para la
  inflación se mantiene sesgado al alza, destacando la depreciación cambiaria
  y presiones en los energéticos." -> 0 (routine risk language, no
  disagreement)
- "Algunos miembros advirtieron que la incertidumbre sobre la trayectoria de
  la inflación ha aumentado." -> +1
- "Tres miembros votaron por mantener la tasa; uno votó por reducirla en 25
  puntos base." -> +1 (one dissent)
- "La decisión se adoptó por mayoría; dos miembros votaron por mantener la
  tasa, argumentando que el panorama es altamente incierto." -> +2
- "Un miembro destacó que los riesgos asociados al tipo de cambio han
  disminuido y que la inflación evoluciona conforme a lo previsto." -> -1

The excerpts are in Spanish. Judge them on meaning, not on specific keywords.
Return only the structured fields."""


MINUTAS_LAX_PROMPT = """\
You are an expert analyst of Banco de México's monetary-policy communication.
You will read short excerpts from the minutes of Banxico's Governing Board
policy meetings (minutas).

How much uncertainty about the economic and monetary-policy outlook does the
excerpt convey? Use your own judgment. Disagreement among board members
counts as uncertainty.

Return:
- relevant: true if the excerpt is about the economic or monetary-policy
  outlook at all; false for tables, figures or administrative text.
- uncertainty_score: from -2 (strongly confident about the outlook) to +2
  (strongly uncertain), 0 if neither; null if not relevant.
- confidence: from 0 to 1, how confident you are in your score.
- evidence: a short Spanish quote (up to 160 characters) behind your score,
  or "" if not relevant."""


# Prompt name -> prompt. The scoring script takes a corpus and a variant and
# looks up "{corpus}_{variant}" (or just "{corpus}" with no variant). A new
# corpus or a new index is a new string above and a new line here.
# Transcripts: paused, no prompt yet.
PROMPTS = {
    "banxico_anuncios_calibrated": ANUNCIOS_CALIBRATED_PROMPT,
    "banxico_anuncios_lax": ANUNCIOS_LAX_PROMPT,
    "banxico_minutas_calibrated": MINUTAS_CALIBRATED_PROMPT,
    "banxico_minutas_lax": MINUTAS_LAX_PROMPT,
}
