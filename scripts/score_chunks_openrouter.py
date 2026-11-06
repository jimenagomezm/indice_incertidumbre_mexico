"""Score chunks with any model on OpenRouter. The step that costs money.

    # what would be sent and what it would cost. Spends nothing.
    uv run python scripts/score_chunks_openrouter.py --pilot --model openai/gpt-6-luna --dry-run

    # the pilot: the same 120 anuncio/minuta chunks for every model and prompt
    uv run python scripts/score_chunks_openrouter.py --pilot --variant calibrated --model openai/gpt-6-luna

    # a second run of the same model, to measure how much it disagrees with itself
    uv run python scripts/score_chunks_openrouter.py --pilot --model openai/gpt-6-luna --run-tag repeat

    # a full corpus
    uv run python scripts/score_chunks_openrouter.py --corpus banxico_minutas --variant lax --model openai/gpt-6-luna

Without --dry-run it shows the estimate and asks you to type SPEND.

What it scores with
-------------------
The prompt is `PROMPTS["{corpus}_{variant}"]` in `src/index/llm_prompts.py`
(just `PROMPTS[corpus]` when no --variant is given). The answer format is `ChunkScore` in `src/index/llm_scoring.py`,
enforced as a strict JSON schema, so a reply is either valid fields or an error.

What it records
---------------
One row per chunk, in `data/processed/banxico/llm_scores/`, named
`{corpus}[__{variant}]__{model}[__pilot][__{run_tag}].parquet`. Besides the scores, each row
keeps the sha256 of the prompt, the model and provider that actually answered,
tokens and the cost OpenRouter charged. Re-running skips chunks already scored
under the same prompt, so an interrupted run resumes and only failed chunks are
retried; edit the prompt and every chunk counts as unscored again.

The least reasoning the model allows, and temperature 0 where the model takes
one (reasoning models do not): a -2..+2 label gains little from hidden
reasoning, which is billed as output and makes answers less repeatable.
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

import paths  # noqa: E402 - must follow the sys.path line above
paths.add_src_subfolders_to_path()

from common import rebuild_parent_chunks  # noqa: E402
from llm_prompts import PROMPTS  # noqa: E402
from llm_scoring import ChunkScore, chunk_score_json_schema  # noqa: E402

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"
API_KEY_NAME = "OPEN_ROUTER_KEY"

SCORES_DIR = paths.PROCESSED_DIR / "llm_scores"

# Room for the JSON answer plus whatever reasoning a model does even at its
# lowest effort. Only the tokens actually produced are billed.
MAX_TOKENS = 2048

# For the estimate only: Spanish runs about 3.6 characters per token, and an
# answer (four fields, a quote of up to 160 characters) about 100 tokens.
CHARS_PER_TOKEN = 3.6
OUTPUT_TOKENS_PER_CALL = 100

# The pilot: 40 anuncio chunks and 80 minuta chunks, drawn once and saved, so
# every model and every repeat scores exactly the same 120. 15 of the minuta
# chunks are drawn from those reporting a split vote, the case the
# disagreement rule in the minutas prompt is about.
PILOT_PATH = SCORES_DIR / "pilot_chunk_ids.csv"
PILOT_SEED = 20261019
PILOT_SIZES = {"banxico_anuncios": 40, "banxico_minutas": 80}
PILOT_DISSENT = 15
DISSENT_PATTERN = r"vot[óo] (?:a favor de|por) |por mayor[ií]a|disid|disent"


def load_parent_chunks(corpus):
    chunks = pd.read_parquet(paths.PROCESSED_DIR / f"{corpus}_chunks.parquet")
    return rebuild_parent_chunks(chunks)


def pilot_chunk_ids():
    """The saved pilot sample; drawn on first use, then never redrawn."""
    if PILOT_PATH.exists():
        return pd.read_csv(PILOT_PATH)

    samples = []
    for corpus, size in PILOT_SIZES.items():
        parents = load_parent_chunks(corpus)
        n_dissent = PILOT_DISSENT if corpus == "banxico_minutas" else 0
        dissent = parents[parents["chunk_text"].str.contains(DISSENT_PATTERN, case=False)]
        picked_dissent = dissent.sample(n_dissent, random_state=PILOT_SEED)
        rest = parents.drop(picked_dissent.index).sample(size - n_dissent, random_state=PILOT_SEED)
        for picked, reason in [(picked_dissent, "dissent"), (rest, "random")]:
            samples.append(pd.DataFrame({"corpus": corpus,
                                         "parent_chunk_id": picked["parent_chunk_id"],
                                         "reason": reason}))

    sample = pd.concat(samples, ignore_index=True)
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_csv(PILOT_PATH, index=False)
    return sample


CORPORA = ["banxico_anuncios", "banxico_minutas"]


def prompt_name(corpus, variant):
    name = f"{corpus}_{variant}" if variant else corpus
    if name not in PROMPTS:
        raise SystemExit(f"no prompt named {name!r} in llm_prompts.PROMPTS")
    return name


def output_path(corpus, variant, model, pilot, run_tag):
    name = f"{corpus}__{variant}" if variant else corpus
    name += f"__{model.replace('/', '_')}"
    if pilot:
        name += "__pilot"
    if run_tag:
        name += f"__{run_tag}"
    return SCORES_DIR / f"{name}.parquet"


def model_info(model):
    """Prices (input, output, USD per million tokens) and the parameters the
    model accepts, from OpenRouter's public list."""
    for entry in httpx.get(MODELS_URL, timeout=60).json()["data"]:
        if entry["id"] == model:
            pricing = entry["pricing"]
            prices = float(pricing["prompt"]) * 1e6, float(pricing["completion"]) * 1e6
            return prices, set(entry.get("supported_parameters", []))
    raise SystemExit(f"{model} is not on OpenRouter's model list - check the id.")


def build_request(model, system_prompt, chunk_text, reasoning_effort, use_temperature):
    request = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": chunk_text}],
        "max_tokens": MAX_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "chunk_score", "strict": True,
                            "schema": chunk_score_json_schema()},
        },
        # Only route to providers that honour every parameter above - otherwise
        # OpenRouter may silently drop the schema or the temperature.
        "provider": {"require_parameters": True},
        "usage": {"include": True},
    }
    # Reasoning models (the GPT-5/6 family) take no temperature at all, and
    # with require_parameters a temperature they do not accept would rule out
    # every provider. So it is sent only where it is accepted.
    if use_temperature:
        request["temperature"] = 0
    if reasoning_effort != "default":
        request["reasoning"] = {"effort": reasoning_effort}
    return request


# A hard limit on one request, connection to last byte. The client's own
# timeout is per read, and OpenRouter keeps a waiting request alive by sending
# filler bytes, so on its own it never fires: in the first full Sol run eight
# requests hung for 16 minutes this way. Past the limit the request is retried.
REQUEST_DEADLINE_SECONDS = 90


def post_with_deadline(client, request):
    """POST, reading the reply in pieces so the deadline can be checked."""
    started = time.monotonic()
    with client.stream("POST", API_URL, json=request) as response:
        pieces = []
        for piece in response.iter_bytes():
            pieces.append(piece)
            if time.monotonic() - started > REQUEST_DEADLINE_SECONDS:
                raise httpx.TimeoutException(f"no complete reply after {REQUEST_DEADLINE_SECONDS}s")
        return response.status_code, json.loads(b"".join(pieces))


def score_one(client, request, attempts=4):
    """One chunk -> a flat record. Never raises: failures come back as `error`."""
    record = {"error": None}
    for attempt in range(attempts):
        try:
            status, body = post_with_deadline(client, request)
            if status in (429, 500, 502, 503, 504) and attempt < attempts - 1:
                time.sleep(2 ** attempt * 5)
                continue
            if "error" in body or status != 200:
                record["error"] = json.dumps(body.get("error", body))[:500]
                return record
            break
        except (httpx.HTTPError, ValueError) as exc:
            if attempt == attempts - 1:
                record["error"] = f"{type(exc).__name__}: {exc}"[:500]
                return record
            time.sleep(2 ** attempt * 5)

    usage = body.get("usage") or {}
    record.update({
        "model_served": body.get("model"),
        "provider": body.get("provider"),
        "generation_id": body.get("id"),
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "cost_usd": usage.get("cost"),
    })

    content = body["choices"][0]["message"].get("content") or ""
    try:
        # Read the first JSON object and ignore anything after it: Luna now
        # and then appends a stray " ." to an otherwise valid answer.
        parsed, _ = json.JSONDecoder().raw_decode(content.strip())
        score = ChunkScore.model_validate(parsed)
    except ValueError:
        record["error"] = f"unparseable reply: {content[:300]}"
        return record
    if score.uncertainty_score is not None and not -2 <= score.uncertainty_score <= 2:
        record["error"] = f"score out of range: {score.uncertainty_score}"
        return record

    record.update(score.model_dump())
    return record


def save(records, done, path):
    """Old rows for chunks that were not re-scored, plus everything new."""
    new = pd.DataFrame(records)
    if len(done):
        new = pd.concat([done[~done["parent_chunk_id"].isin(new["parent_chunk_id"])], new],
                        ignore_index=True)
    new.sort_values("parent_chunk_id").to_parquet(path, index=False)


def run_corpus(args, corpus, ids, prices):
    system_prompt = PROMPTS[prompt_name(corpus, args.variant)]
    prompt_hash = sha256(system_prompt.encode("utf-8")).hexdigest()
    parents = load_parent_chunks(corpus)
    if ids is not None:
        parents = parents[parents["parent_chunk_id"].isin(ids)]

    path = output_path(corpus, args.variant, args.model, args.pilot, args.run_tag)
    done = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["parent_chunk_id"])
    scored = set()
    if len(done):
        ok = done["error"].isna() & (done["prompt_sha256"] == prompt_hash)
        scored = set(done.loc[ok, "parent_chunk_id"])
    pending = parents[~parents["parent_chunk_id"].isin(scored)].sort_values("parent_chunk_id")

    input_tokens = (pending["chunk_text"].str.len().sum()
                    + len(pending) * len(system_prompt)) / CHARS_PER_TOKEN
    output_tokens = len(pending) * OUTPUT_TOKENS_PER_CALL
    estimate = (input_tokens * prices[0] + output_tokens * prices[1]) / 1e6

    print(f"\n{corpus}  ->  {path.relative_to(PROJECT_DIR)}")
    print(f"  chunks: {len(parents)}, already scored: {len(scored)}, to score: {len(pending)}")
    print(f"  prompt sha256: {prompt_hash[:12]}")
    print(f"  estimate: ~{input_tokens / 1e6:.2f}M tokens in, ~{output_tokens / 1e6:.2f}M out"
          f"  ->  ~${estimate:.2f}  (more if the model reasons)")
    return pending, done, path, system_prompt, prompt_hash, estimate


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="OpenRouter id, e.g. openai/gpt-6-luna")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--corpus", nargs="+", choices=CORPORA)
    which.add_argument("--pilot", action="store_true", help="the saved 120-chunk pilot sample")
    parser.add_argument("--variant", default="",
                        help="prompt variant, e.g. calibrated or lax (see llm_prompts.PROMPTS)")
    parser.add_argument("--run-tag", default="", help="suffix for a repeat run, e.g. 'repeat'")
    parser.add_argument("--reasoning-effort", default="none",
                        help="none / minimal / low / medium / high, or 'default' to send nothing. "
                             "Some models refuse 'none' (gpt-6.1-sol: use minimal); one that "
                             "ignores a level it does not know may reason at length instead "
                             "(mistral-large-4 with 'minimal' ran out of tokens)")
    parser.add_argument("--max-cost", type=float, default=5.0,
                        help="stop sending once this run has cost this many USD")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prices, supported = model_info(args.model)
    use_temperature = "temperature" in supported
    print(f"{args.model}: ${prices[0]:.2f} in / ${prices[1]:.2f} out per million tokens"
          f"; temperature 0: {'yes' if use_temperature else 'not accepted by this model'}")

    if args.pilot:
        sample = pilot_chunk_ids()
        targets = [(corpus, set(group["parent_chunk_id"])) for corpus, group in sample.groupby("corpus")]
    else:
        targets = [(corpus, None) for corpus in args.corpus]

    plans = [(corpus, *run_corpus(args, corpus, ids, prices)) for corpus, ids in targets]
    total = sum(plan[6] for plan in plans)
    print(f"\ntotal estimate: ~${total:.2f}   (stops at ${args.max_cost:.2f})")

    if args.dry_run:
        corpus, pending = plans[0][0], plans[0][1]
        if len(pending):
            request = build_request(args.model, "<system prompt>", pending["chunk_text"].iloc[0][:300] + "...",
                                    args.reasoning_effort, use_temperature)
            print(f"\nexample request ({corpus}):\n{json.dumps(request, ensure_ascii=False, indent=2)[:1500]}")
        print("\ndry run: nothing sent.")
        return

    if input("\nType SPEND to send these requests: ").strip() != "SPEND":
        print("Nothing sent.")
        return

    api_key = os.environ.get(API_KEY_NAME, "").strip()
    if not api_key:
        raise SystemExit(f"{API_KEY_NAME} is not set in .env")
    client = httpx.Client(timeout=120, headers={"Authorization": f"Bearer {api_key}"})

    # One call first: a parameter the model rejects then costs one failed
    # request, not a whole run of them.
    _, pending, _, _, system_prompt, *_rest = next(plan for plan in plans if len(plan[1]))
    test = score_one(client, build_request(args.model, system_prompt, pending["chunk_text"].iloc[0],
                                           args.reasoning_effort, use_temperature))
    if test["error"] is not None:
        raise SystemExit(f"test call failed, nothing else sent:\n{test['error']}")
    print(f"test call ok: served by {test['provider']} ({test['model_served']}), "
          f"{test['output_tokens']} output tokens ({test['reasoning_tokens']} reasoning), ${test['cost_usd']}")

    spent = test["cost_usd"] or 0.0
    for corpus, pending, done, path, system_prompt, prompt_hash, _ in plans:
        records, stopped = [], False
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for row in pending.itertuples(index=False):
                request = build_request(args.model, system_prompt, row.chunk_text,
                                        args.reasoning_effort, use_temperature)
                futures[pool.submit(score_one, client, request)] = row
            for n, future in enumerate(as_completed(futures), start=1):
                row = futures[future]
                record = future.result()
                record.update({
                    "parent_chunk_id": row.parent_chunk_id,
                    "document_id": row.document_id,
                    "model_requested": args.model,
                    "reasoning_effort": args.reasoning_effort,
                    "temperature_0": use_temperature,
                    "prompt_sha256": prompt_hash,
                    "scored_at": datetime.now(timezone.utc).isoformat(),
                })
                records.append(record)
                spent += record.get("cost_usd") or 0.0
                if record["error"] is not None:
                    print(f"  error on {row.parent_chunk_id}: {record['error'][:150]}")
                if n % 50 == 0:
                    save(records, done, path)
                    print(f"  {n}/{len(futures)}  spent ${spent:.2f}")
                if spent >= args.max_cost and not stopped:
                    stopped = True
                    for other in futures:
                        other.cancel()
                    print(f"  reached --max-cost ${args.max_cost:.2f}; stopping.")
        if records:
            save(records, done, path)
        errors = sum(r["error"] is not None for r in records)
        print(f"{corpus}: {len(records)} scored, {errors} errors, total spent ${spent:.2f}")
        if errors:
            first = next(r["error"] for r in records if r["error"] is not None)
            print(f"  first error: {first}")
        if stopped:
            break


if __name__ == "__main__":
    main()
