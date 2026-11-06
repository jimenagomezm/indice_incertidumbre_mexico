"""Pull economic series from Banxico's SIE API (Sistema de Información
Económica) for validating text-based indices against real data.

This does NOT touch the text corpus. It fetches numeric series (e.g. the
policy/target rate) used only as an external "answer key": we check whether
our text-derived index moves sensibly with real monetary-policy dynamics.

Requires a free API token. Register at:
    https://www.banxico.org.mx/SieAPIRest/service/v1/token
then put it in a .env file at the project root (which is gitignored):
    BANXICO_SIE_TOKEN=your_token_here

Same request-based pattern as src/scraping.py; reusable for Phase B
(Hawkish/Dovish stance vs. actual rate decisions).
"""

import os
from pathlib import Path

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is a declared dep, but degrade gracefully
    load_dotenv = None

SIE_BASE_URL = "https://www.banxico.org.mx/SieAPIRest/service/v1"

# Default series of interest. IDs are SIE series identifiers; verify against
# the SIE catalog before relying on them for analysis.
#   SF61745 — Tasa objetivo (overnight interbank target rate), the policy rate.
# Two external series, measuring different things on purpose.
#
#   inflation_forecast_dispersion - the standard deviation, across private-sector
#     forecasters, of expected inflation over the next 12 months. This is the
#     PRIMARY check: it is a direct measure of how much disagreement (and so
#     uncertainty) there is about the outlook, which is the construct the index
#     claims to measure. Monthly, and it starts in 1999, so it covers the whole
#     corpus.
#
#   exchange_rate - the daily FIX peso/dollar rate. Its realized volatility is
#     the classic market-based uncertainty barometer for Mexico, needs no survey,
#     and runs from 1991, so it covers the whole corpus with room to spare. This
#     is the PRIMARY benchmark.
#
#   target_rate - the overnight interbank target rate. Differenced and taken in
#     absolute value it gives policy *activity*: how much the bank is actually
#     moving. That is a different question - "does the index anticipate policy
#     action?" - and a useful one, but it is not uncertainty. It also only starts
#     in 2008, so a third of the corpus has no overlap with it.
DEFAULT_SERIES = {
    "inflation_forecast_dispersion": "SR14200",
    "target_rate": "SF61745",
    "exchange_rate": "SF43718",
}


def get_sie_token(token=None):
    """Return the SIE API token, from the argument or the environment/.env."""
    if token:
        return token
    if load_dotenv is not None:
        load_dotenv()  # loads a project-root .env if present
    token = os.environ.get("BANXICO_SIE_TOKEN")
    if not token:
        raise RuntimeError(
            "No SIE token found. Register at "
            "https://www.banxico.org.mx/SieAPIRest/service/v1/token and set "
            "BANXICO_SIE_TOKEN in a .env file at the project root."
        )
    return token


def fetch_sie_series(series_ids, token=None, start=None, end=None, timeout=60):
    """Fetch one or more SIE series and return a tidy long DataFrame.

    Parameters
    ----------
    series_ids : str | list[str]
        One or more SIE series identifiers (e.g. "SF61745").
    token : str, optional
        SIE API token; falls back to BANXICO_SIE_TOKEN in the environment/.env.
    start, end : str, optional
        Date range as "YYYY-MM-DD". If omitted, the full history is returned.

    Returns
    -------
    DataFrame with columns: series_id, date (datetime), value (float).
    Missing observations (SIE returns "N/E") become NaN.
    """
    if isinstance(series_ids, str):
        series_ids = [series_ids]
    token = get_sie_token(token)

    ids = ",".join(series_ids)
    if start and end:
        # SIE expects dd/mm/yyyy in the path
        start_fmt = pd.Timestamp(start).strftime("%Y-%m-%d")
        end_fmt = pd.Timestamp(end).strftime("%Y-%m-%d")
        url = f"{SIE_BASE_URL}/series/{ids}/datos/{start_fmt}/{end_fmt}"
    else:
        url = f"{SIE_BASE_URL}/series/{ids}/datos"

    response = requests.get(url, headers={"Bmx-Token": token}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    records = []
    for serie in payload.get("bmx", {}).get("series", []):
        series_id = serie.get("idSerie")
        for observation in serie.get("datos", []):
            raw_value = observation.get("dato")
            # SIE marks missing values as "N/E"; commas are thousands separators
            value = pd.to_numeric(str(raw_value).replace(",", ""), errors="coerce")
            records.append(
                {
                    "series_id": series_id,
                    "date": pd.to_datetime(observation.get("fecha"), format="%d/%m/%Y", errors="coerce"),
                    "value": value,
                }
            )

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values(["series_id", "date"]).reset_index(drop=True)
    return result


def load_series_cached(name, cache_dir, token=None, refresh=False):
    """One of DEFAULT_SERIES, from a local cache when possible.

    The external check is the only evidence we have about whether the *scores*
    are right, so it should not depend on whoever runs it holding an SIE token.
    Each series is a few hundred rows: fetch once, commit the cache, and the
    numbers reproduce offline afterwards.

    Pass refresh=True to re-download (for example to extend a series with newly
    published months).
    """
    if name not in DEFAULT_SERIES:
        raise KeyError(f"unknown series {name!r}; known: {sorted(DEFAULT_SERIES)}")

    cache_path = Path(cache_dir) / f"sie_{name}.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    series = fetch_sie_series(DEFAULT_SERIES[name], token=token)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    series.to_parquet(cache_path, index=False)
    return series
