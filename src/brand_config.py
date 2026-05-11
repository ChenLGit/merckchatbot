# src/brand_config.py
"""
Brand profile registry + active-brand selector.

This module is the single source of truth for ANY brand-specific text,
color, column, or competitor list used anywhere else in the application.
The rest of the codebase (prompts, planner, agent graph, news engine,
visualizations, Streamlit UI) reads from `BRAND` instead of hard-coding
"Opdivo" or "Keytruda".

Two production-ready profiles are shipped:

  - "bms_opdivo"     : Bristol Myers Squibb marketing team for Opdivo
                       (nivolumab). Default.
  - "merck_keytruda" : Merck & Co. marketing team for Keytruda
                       (pembrolizumab). The original brand the app was
                       built for; preserved here so we can swap back at
                       any time.

The underlying targeting model output (`pred_class`, `pred_proba`, SHAP
drivers in `data/raw/MerckAI_table.csv`) is brand-agnostic and is reused
as-is for whichever brand is active. The CSV already contains per-drug
billing-share columns for both Opdivo and Keytruda, so each brand
profile points at its own "our drug" column without any data change.

How to switch brands:
  1. Streamlit Cloud / `.streamlit/secrets.toml`:
         ACTIVE_BRAND = "merck_keytruda"
  2. Local shell:
         set ACTIVE_BRAND=merck_keytruda     (Windows)
         export ACTIVE_BRAND=merck_keytruda  (macOS / Linux)
  3. Or edit `DEFAULT_BRAND_KEY` below.

No code change is required to flip — every layer re-derives its
brand-specific text from this module at import time.
"""

from __future__ import annotations

import os


# =============================================================================
# Competitor metadata (shared building blocks)
# =============================================================================
# Schema matches what `news_engine.COMPETITORS` historically used:
#   brand / generic / company / column_token / aliases / match_tokens

_KEYTRUDA_AS_COMPETITOR = {
    "brand": "Keytruda",
    "generic": "pembrolizumab",
    "company": "Merck & Co.",
    "column_token": "hcpcs_category_keytruda_br_pct",
    "aliases": ["keytruda", "merck", "pembrolizumab", "pembro"],
    "match_tokens": ["keytruda", "pembrolizumab", "merck"],
}

_OPDIVO_AS_COMPETITOR = {
    "brand": "Opdivo",
    "generic": "nivolumab",
    "company": "Bristol-Myers Squibb",
    "column_token": "hcpcs_category_opdivo_br_pct",
    "aliases": ["opdivo", "bms", "bristol", "nivolumab"],
    "match_tokens": [
        "opdivo", "nivolumab", "bristol-myers", "bristol myers", "bms",
    ],
}

_TECENTRIQ = {
    "brand": "Tecentriq",
    "generic": "atezolizumab",
    "company": "Roche / Genentech",
    "column_token": "hcpcs_category_tecentriq_br_pct",
    "aliases": ["tecentriq", "roche", "genentech", "atezolizumab"],
    "match_tokens": ["tecentriq", "atezolizumab", "genentech", "roche"],
}

_IMFINZI = {
    "brand": "Imfinzi",
    "generic": "durvalumab",
    "company": "AstraZeneca",
    "column_token": "hcpcs_category_imfinzi_br_pct",
    "aliases": ["imfinzi", "astrazeneca", "durvalumab"],
    "match_tokens": ["imfinzi", "durvalumab", "astrazeneca", "astra zeneca"],
}

_LIBTAYO = {
    "brand": "Libtayo",
    "generic": "cemiplimab",
    "company": "Regeneron / Sanofi",
    "column_token": "hcpcs_category_libtayo_br_pct",
    "aliases": ["libtayo", "regeneron", "sanofi", "cemiplimab"],
    "match_tokens": ["libtayo", "cemiplimab", "regeneron", "sanofi"],
}


# =============================================================================
# Brand registry
# =============================================================================
BRANDS: dict[str, dict] = {
    "bms_opdivo": {
        "key": "bms_opdivo",
        # Display / identity
        "company": "Bristol Myers Squibb",
        "company_short": "BMS",
        "drug": "Opdivo",
        "drug_generic": "nivolumab",
        "display": "Bristol Myers Squibb · Opdivo",
        "page_title": "Bristol Myers Squibb Data Science Hub",
        "subtitle": "Provider Targeting & Marketing Strategy AI Application",
        # Visuals
        "color_primary": "#BE2BBB",  # BMS corporate purple
        # Data
        "our_drug_column_token": "hcpcs_category_opdivo_br_pct",
        "our_drug_share_label": "Opdivo (Bristol Myers Squibb)",
        # Network identity for outbound article fetches
        "user_agent": "BMSOpdivoBot/1.0",
        "user_agent_url": "+https://bms.com",
        # Competitors of OUR drug (Keytruda is in this set, Opdivo is NOT).
        "competitors": {
            "keytruda": _KEYTRUDA_AS_COMPETITOR,
            "tecentriq": _TECENTRIQ,
            "imfinzi": _IMFINZI,
            "libtayo": _LIBTAYO,
        },
    },
    "merck_keytruda": {
        "key": "merck_keytruda",
        # Display / identity
        "company": "Merck & Co.",
        "company_short": "Merck",
        "drug": "Keytruda",
        "drug_generic": "pembrolizumab",
        "display": "Merck Keytruda",
        "page_title": "Merck Data Science Hub",
        "subtitle": "Provider Targeting Strategy AI Application",
        # Visuals
        "color_primary": "#00857c",  # Merck heritage green
        # Data
        "our_drug_column_token": "hcpcs_category_keytruda_br_pct",
        "our_drug_share_label": "Keytruda (Merck)",
        # Network identity for outbound article fetches
        "user_agent": "MerckKeytrudaBot/1.0",
        "user_agent_url": "+https://merck.com",
        # Competitors of OUR drug (Opdivo is in this set, Keytruda is NOT).
        "competitors": {
            "opdivo": _OPDIVO_AS_COMPETITOR,
            "tecentriq": _TECENTRIQ,
            "imfinzi": _IMFINZI,
            "libtayo": _LIBTAYO,
        },
    },
}

DEFAULT_BRAND_KEY = "bms_opdivo"


# =============================================================================
# Active-brand selection
# =============================================================================
def _read_active_brand_key() -> str:
    """Resolve the active brand key from (in priority order):
        1. Streamlit secrets `ACTIVE_BRAND`
        2. Environment variable `ACTIVE_BRAND`
        3. `DEFAULT_BRAND_KEY`
    Falls through silently on any error so non-Streamlit importers
    (CLI scripts, tests, `scripts/render_graph.py`) still work.
    """
    # 1. Streamlit secrets, if Streamlit is importable AND the secrets
    #    file has been initialized. Any failure -> fall through.
    try:
        import streamlit as st  # noqa: WPS433
        val = st.secrets.get("ACTIVE_BRAND", None)  # type: ignore[attr-defined]
        if val:
            return str(val).strip().lower()
    except Exception:  # noqa: BLE001
        pass

    # 2. Environment variable.
    env_val = os.environ.get("ACTIVE_BRAND")
    if env_val:
        return env_val.strip().lower()

    # 3. Default.
    return DEFAULT_BRAND_KEY


def get_brand(key: str | None = None) -> dict:
    """Return the brand profile for `key`, or the active brand if omitted."""
    k = (key or _read_active_brand_key()).strip().lower()
    if k not in BRANDS:
        k = DEFAULT_BRAND_KEY
    return BRANDS[k]


# Resolve once at import. Anything that needs the active brand at module
# load time (e.g. `prompts.py` building system-prompt strings, `planner.py`
# building the tool enum) reads from `BRAND` directly.
ACTIVE_BRAND_KEY: str = _read_active_brand_key()
if ACTIVE_BRAND_KEY not in BRANDS:
    ACTIVE_BRAND_KEY = DEFAULT_BRAND_KEY
BRAND: dict = BRANDS[ACTIVE_BRAND_KEY]


# =============================================================================
# Helpers used across prompts / planner / agent_graph
# =============================================================================
def competitor_keys(brand: dict | None = None) -> list[str]:
    """Lower-cased competitor keys for the active brand (planner enum, etc.)."""
    b = brand or BRAND
    return list(b["competitors"].keys())


def competitor_brand_names(brand: dict | None = None) -> list[str]:
    """Display names of competitors: ['Keytruda', 'Tecentriq', ...]."""
    b = brand or BRAND
    return [c["brand"] for c in b["competitors"].values()]


def competitor_brand_list(brand: dict | None = None) -> str:
    """Inline competitor list for prompts.

    e.g. 'Keytruda (Merck), Tecentriq (Roche), Imfinzi (AstraZeneca), Libtayo (Regeneron)'
    """
    b = brand or BRAND
    return ", ".join(
        f"{c['brand']} ({c['company'].split('/')[0].strip()})"
        for c in b["competitors"].values()
    )


def competitor_brand_bullets(
    brand: dict | None = None,
    indent: str = "    ",
) -> str:
    """Indented bullet block listing each competitor with generic + maker.

    e.g.
        * Keytruda (pembrolizumab, Merck & Co.)
        * Tecentriq (atezolizumab, Roche / Genentech)
        ...
    """
    b = brand or BRAND
    return "\n".join(
        f"{indent}* {c['brand']} ({c['generic']}, {c['company']})"
        for c in b["competitors"].values()
    )


def competitor_match_tokens(brand: dict | None = None) -> tuple[str, ...]:
    """Flat, deduplicated tuple of all competitor match-tokens.

    Used by the LangGraph competitor-claim guardrail to decide whether
    a draft answer mentions a competitor at all (cheap short-circuit
    before invoking the LLM-as-judge).
    """
    b = brand or BRAND
    seen: set[str] = set()
    out: list[str] = []
    for comp in b["competitors"].values():
        for tok in comp.get("match_tokens", []):
            t = (tok or "").lower().strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
    return tuple(out)
