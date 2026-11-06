"""The answer format for Methodology 3 (LLM).

Every prompt in `llm_prompts.py` asks the model for the same four fields.
`ChunkScore` defines them, and `scripts/score_chunks_openrouter.py` sends this
schema to the API as a strict JSON schema, so every reply is either these
fields with valid types or an error. Nothing has to be pulled out of free text.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict


class ChunkScore(BaseModel):
    # extra="forbid" makes pydantic emit "additionalProperties": false in the
    # JSON schema, which the structured-output format requires.
    model_config = ConfigDict(extra="forbid")

    relevant: bool                    # is the chunk about the macro / monetary-policy outlook?
    uncertainty_score: Optional[int]  # -2 (certainty) .. +2 (uncertainty); None if not relevant
    confidence: float                 # 0..1, the model's confidence in its score
    evidence: str                     # short Spanish quote behind the score, or ""


def chunk_score_json_schema():
    """The ChunkScore schema as the JSON the API's structured-output format takes."""
    return ChunkScore.model_json_schema()
