# src/news_engine.py
"""
NEWS intent engine.

Combines two signals so the LLM can answer questions like
"give me top competitor's movement in NJ and what should I do for marketing":

1. Internal (MerckAI_table.csv): per-state average billing share for Keytruda
   and each competitor drug, plus the top-propensity HCP in that state from
   the existing RAG engine (so marketing advice is grounded in a real NPI).
2. External (DuckDuckGo news search, no API key): recent articles about the
   competitor drugs / manufacturers.

Returns a plain dict so the Streamlit layer can format and persist it.
"""

from __future__ import annotations

import pandas as pd

from .rag_engine import _infer_state_from_query, get_hcp_scorecard


# -----------------------------------------------------------------------------
# Competitor metadata
# -----------------------------------------------------------------------------
# The CSV has slightly inconsistent column names (e.g. "opdivo_br   _pct"
# with extra spaces), so we match columns by stripped-lowercase tokens.
COMPETITORS = {
    "opdivo": {
        "brand": "Opdivo",
        "generic": "nivolumab",
        "company": "Bristol-Myers Squibb",
        "column_token": "hcpcs_category_opdivo_br_pct",
        "aliases": ["opdivo", "bms", "bristol", "nivolumab"],
        # Tokens that, if present in the result's title or body, mark the
        # article as actually about this drug. Prevents generic-"competitor"
        # noise (e.g. sports competitors) from leaking into results.
        "match_tokens": ["opdivo", "nivolumab", "bristol-myers", "bristol myers", "bms"],
    },
    "tecentriq": {
        "brand": "Tecentriq",
        "generic": "atezolizumab",
        "company": "Roche / Genentech",
        "column_token": "hcpcs_category_tecentriq_br_pct",
        "aliases": ["tecentriq", "roche", "genentech", "atezolizumab"],
        "match_tokens": ["tecentriq", "atezolizumab", "genentech", "roche"],
    },
    "imfinzi": {
        "brand": "Imfinzi",
        "generic": "durvalumab",
        "company": "AstraZeneca",
        "column_token": "hcpcs_category_imfinzi_br_pct",
        "aliases": ["imfinzi", "astrazeneca", "durvalumab"],
        "match_tokens": ["imfinzi", "durvalumab", "astrazeneca", "astra zeneca"],
    },
    "libtayo": {
        "brand": "Libtayo",
        "generic": "cemiplimab",
        "company": "Regeneron / Sanofi",
        "column_token": "hcpcs_category_libtayo_br_pct",
        "aliases": ["libtayo", "regeneron", "sanofi", "cemiplimab"],
        "match_tokens": ["libtayo", "cemiplimab", "regeneron", "sanofi"],
    },
}

MERCK_COLUMN_TOKEN = "hcpcs_category_keytruda_br_pct"


def _resolve_column(df: pd.DataFrame, token: str) -> str | None:
    """Return the real column name whose stripped-lowercase form matches `token`."""
    token = token.strip().lower().replace(" ", "")
    for col in df.columns:
        if col.strip().lower().replace(" ", "") == token:
            return col
    return None


def detect_competitors(query: str) -> list[str]:
    """Return the competitor keys mentioned in the query, or all four if none match."""
    q = (query or "").lower()
    hits = [key for key, meta in COMPETITORS.items() if any(a in q for a in meta["aliases"])]
    return hits or list(COMPETITORS.keys())


def compute_state_share(df: pd.DataFrame, state: str | None, competitor_keys: list[str]) -> dict:
    """
    Average billing-share (mean %) for Keytruda and each competitor in the
    given state (or full dataset if state is None). Values are 0-100 scale
    if the CSV stores percents, or 0-1 if stored as proportions — we just
    surface them as-is and let the LLM reason about relative magnitude.
    """
    share: dict[str, float] = {}

    state_col_name = "Rndrng_Prvdr_State_Abrvtn"
    if state and state_col_name in df.columns:
        mask = df[state_col_name].astype(str).str.strip().str.upper() == state
        scope = df[mask]
    else:
        scope = df

    if scope.empty:
        return share

    merck_col = _resolve_column(df, MERCK_COLUMN_TOKEN)
    if merck_col is not None:
        share["Keytruda (Merck)"] = float(pd.to_numeric(scope[merck_col], errors="coerce").mean())

    for key in competitor_keys:
        meta = COMPETITORS[key]
        col = _resolve_column(df, meta["column_token"])
        if col is None:
            continue
        label = f"{meta['brand']} ({meta['company']})"
        share[label] = float(pd.to_numeric(scope[col], errors="coerce").mean())

    return share


# -----------------------------------------------------------------------------
# Live web search (DuckDuckGo, no API key)
# -----------------------------------------------------------------------------
def _get_ddgs_cls():
    """Try the newer `ddgs` package first, then fall back to the legacy one."""
    try:
        from ddgs import DDGS  # type: ignore
        return DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore
            return DDGS
        except ImportError:
            return None


def search_news(queries: list[str], max_per_query: int = 3, timelimit: str = "m") -> list[dict] | None:
    """
    Run each query against DuckDuckGo News. Returns a list of dicts or None
    if the search library / network is unavailable (caller should handle).
    `timelimit`: 'd' day, 'w' week, 'm' month, 'y' year.
    """
    DDGS = _get_ddgs_cls()
    if DDGS is None:
        return None

    results: list[dict] = []
    try:
        with DDGS() as ddgs:
            for q in queries:
                try:
                    hits = list(ddgs.news(q, max_results=max_per_query, timelimit=timelimit))
                except Exception:
                    hits = []
                for h in hits:
                    results.append({
                        "query": q,
                        "title": h.get("title"),
                        "source": h.get("source"),
                        "date": h.get("date"),
                        "url": h.get("url") or h.get("href"),
                        "body": h.get("body") or h.get("excerpt"),
                    })
    except Exception:
        return None
    return results


def _build_competitor_queries(competitor_key: str) -> list[str]:
    """Generate a handful of drug-specific search strings for DuckDuckGo.

    We deliberately avoid generic English words like 'competitor' or
    'movement', which cause DDG to surface unrelated content (sports
    competitors, market-movement articles about unrelated tickers, etc.)
    when the drug itself has few recent hits.
    """
    meta = COMPETITORS[competitor_key]
    brand = meta["brand"]
    generic = meta.get("generic") or ""
    company = meta["company"].split("/")[0].strip()  # primary company only
    queries = [
        f"{brand} {generic}".strip(),
        f"{brand} {company} oncology",
        f"{brand} FDA approval",
        f"{brand} clinical trial",
    ]
    # De-duplicate while preserving order.
    seen = set()
    uniq = []
    for q in queries:
        key = q.lower().strip()
        if key and key not in seen:
            seen.add(key)
            uniq.append(q)
    return uniq


def _item_matches_competitor(item: dict, match_tokens: list[str]) -> bool:
    """True if the article title or body contains at least one token that
    identifies the target drug (brand, generic, or manufacturer)."""
    if not match_tokens:
        return True
    haystack = " ".join([
        str(item.get("title") or ""),
        str(item.get("body") or ""),
    ]).lower()
    return any(tok.lower() in haystack for tok in match_tokens)


def _dedupe_news(items: list[dict]) -> list[dict]:
    """Drop duplicates by URL, preserving first occurrence."""
    seen = set()
    out = []
    for it in items:
        key = (it.get("url") or it.get("title") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def search_competitor_news(
    competitor_keys: list[str],
    max_per_competitor: int = 3,
) -> list[dict] | None:
    """
    Run drug-specific DDG searches for each competitor key and aggressively
    filter the results down to items that actually mention the drug. Falls
    back from last-month to last-year if the tight window returns nothing.
    Returns None only if the search library / network is totally unavailable.
    """
    if not competitor_keys:
        return []

    all_kept: list[dict] = []
    any_library_available = False

    for key in competitor_keys:
        meta = COMPETITORS[key]
        queries = _build_competitor_queries(key)

        for timelimit in ("m", "y"):  # try last month first, then last year
            raw = search_news(queries, max_per_query=max_per_competitor, timelimit=timelimit)
            if raw is None:
                # library/network unavailable -> bail out for this competitor
                break
            any_library_available = True
            filtered = [
                r for r in raw
                if _item_matches_competitor(r, meta.get("match_tokens") or [])
            ]
            if filtered:
                # Tag which competitor this batch belongs to (useful for debugging).
                for r in filtered:
                    r["_competitor"] = key
                all_kept.extend(filtered)
                break  # done for this competitor

    if not any_library_available:
        return None

    return _dedupe_news(all_kept)


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------
def get_competitor_brief(query: str, df: pd.DataFrame, max_news_per_competitor: int = 3) -> dict:
    """
    Put together everything the NEWS prompt needs:
      - inferred `state` (or None -> national)
      - `competitors`: list of keys relevant to the query
      - `share`: state-level billing mix from the CSV
      - `news`: list of web-search news items (may be [] or None on failure)
      - `top_hcp`: the same top-propensity provider the OPPORTUNITY intent
                   would have surfaced, so marketing advice has a real NPI.
    """
    state = _infer_state_from_query(query, df)
    competitor_keys = detect_competitors(query)
    share = compute_state_share(df, state, competitor_keys)

    # Only search the first 3 detected competitors to keep latency reasonable.
    news_items = search_competitor_news(
        competitor_keys[:3],
        max_per_competitor=max_news_per_competitor,
    )

    top_hcp = get_hcp_scorecard(query, df)

    return {
        "state": state,
        "competitors": competitor_keys,
        "share": share,
        "news": news_items,
        "top_hcp": top_hcp,
    }
