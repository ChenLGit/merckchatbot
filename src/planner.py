# src/planner.py
"""
LLM tool-calling planner (Option 2).

Replaces the regex/priority-list routing in `src/router.py` with a single
Groq chat completion that uses tool / function calling to:

  1. Understand the user's (possibly compound) query.
  2. Extract structured arguments (state, NPI, competitor keys, headline
     index) so downstream handlers don't have to re-infer them.
  3. Emit an ORDERED list of tool calls to run. Multiple tools in one turn
     are supported, which replaces the hand-rolled `intent_queue` +
     yes/no confirmation state machine.

This module is INTENTIONALLY SIDE-EFFECT FREE: it only decides *what* to do.
The Streamlit layer still owns dispatch, rendering, and session state, so
we can trivially fall back to the legacy router (`src.router.get_intents`)
if anything here fails.
"""

from __future__ import annotations

import json
import traceback
from typing import Any

from groq import Groq


# Populated by `plan_actions` on a hard failure so the caller (and the UI)
# can surface the actual exception instead of a generic "planner unavailable".
# Cleared on every successful call.
LAST_ERROR: dict[str, str] | None = None


# -----------------------------------------------------------------------------
# Tool schemas (OpenAI-compatible, consumed by Groq `tools=` parameter)
# -----------------------------------------------------------------------------
# Each tool corresponds to one existing handler in streamlit_app.py. The
# `name` is the dispatch key; `parameters` are what the planner extracts
# from the user's query.
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_hcp_opportunity",
            "description": (
                "Identify ONE specific high-propensity HCP and return their "
                "opportunity scorecard (location, propensity score, model "
                "drivers). Use when the user asks about a SPECIFIC provider, "
                "the TOP provider in a state, who to prioritize, or looks "
                "up an explicit 10-digit NPI. "
                "DO NOT use this tool for aggregate / counting / "
                "statistical questions like 'how many providers are in our "
                "dataset', 'how many states do we cover', 'what specialties "
                "do we track', or 'what's a typical HCP profile'. Those "
                "are definitional questions that belong to `general_advisor`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {
                        "type": ["string", "null"],
                        "description": (
                            "Two-letter USPS state code (e.g. 'NJ', 'TX') if "
                            "the user named a state. Omit if unspecified."
                        ),
                    },
                    "npi": {
                        "type": ["string", "null"],
                        "description": (
                            "Exact 10-digit NPI if the user quoted one. "
                            "Omit if unspecified."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_marketing_strategy",
            "description": (
                "Generate a Next-Best-Action (NBA) marketing strategy for a "
                "target HCP, grounded in their digital-adoption score, "
                "preferred channel, and last-engagement recency. Use when "
                "the user asks about marketing, outreach, channels, "
                "timing, engagement, or 'how should we approach / market to "
                "this HCP'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {"type": ["string", "null"]},
                    "npi": {"type": ["string", "null"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_competitor_news",
            "description": (
                "Fetch recent competitor news (Opdivo, Tecentriq, Imfinzi, "
                "Libtayo) and the internal billing-share mix for the "
                "requested scope. Use for any RECENT-EVENT question about "
                "competitor movement, market share shifts, FDA approvals, "
                "clinical-trial readouts, label changes, or pipeline deals. "
                "Do NOT use for purely definitional questions like 'who are "
                "our competitors' — use general_advisor for those."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {
                        "type": ["string", "null"],
                        "description": (
                            "Two-letter USPS state code to scope the brief. "
                            "Omit for a national view."
                        ),
                    },
                    "competitors": {
                        "type": ["array", "null"],
                        "description": (
                            "Subset of competitors the user explicitly "
                            "named. Omit to cover all four."
                        ),
                        "items": {
                            "type": "string",
                            "enum": [
                                "opdivo",
                                "tecentriq",
                                "imfinzi",
                                "libtayo",
                            ],
                        },
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_article",
            "description": (
                "Summarize ONE of the previously shown news headlines. Only "
                "use this if the user has already been shown a list of "
                "headlines AND is now asking about a specific one (e.g. "
                "'tell me more about #2', 'summarize the Opdivo FDA "
                "article', 'deep dive on the first headline'). The caller "
                "will provide the cached headline list with indices in the "
                "system message."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "headline_index": {
                        "type": "integer",
                        "description": (
                            "Zero-based index into the cached headlines list "
                            "provided in the system message."
                        ),
                    },
                },
                "required": ["headline_index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "general_advisor",
            "description": (
                "Answer a general Keytruda / oncology / PD-1 / commercial "
                "strategy question that does NOT require a single-HCP "
                "lookup, marketing-NBA calculation, or recent-news search. "
                "Use for: (a) definitional questions ('who are our "
                "competitors?', 'what is PD-1?', 'how does our propensity "
                "model work?'); (b) aggregate / counting / statistical "
                "questions about the targeting dataset ('how many providers "
                "are in our dataset', 'how many states do we cover', "
                "'what specialties do we track', 'what's a typical HCP "
                "profile'); (c) mechanism-of-action, market-access theory, "
                "strategy overviews; (d) meta questions about this "
                "application."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": (
                            "The question to answer. Usually a verbatim "
                            "echo of the user's input."
                        ),
                    },
                },
            },
        },
    },
]


# Canonical set of tool names we are willing to dispatch to. Anything the
# model hallucinates outside this set is dropped.
VALID_TOOL_NAMES: set[str] = {t["function"]["name"] for t in TOOLS}


# -----------------------------------------------------------------------------
# Planner system prompt
# -----------------------------------------------------------------------------
PLANNER_SYSTEM_PROMPT = """
You are the Merck Keytruda Strategy Planner.

Ground truth (never violate):
- Keytruda (pembrolizumab) is OUR drug (Merck). It is NEVER a competitor.
- Keytruda's direct competitors are Opdivo (BMS), Tecentriq (Roche /
  Genentech), Imfinzi (AstraZeneca), and Libtayo (Regeneron / Sanofi).

Your job is to decide which internal tools to call to answer the user's
question. You have five tools available:

  - get_hcp_opportunity    : HCP targeting / top providers / NPI lookup.
  - get_marketing_strategy : Per-HCP Next-Best-Action, channel, timing.
  - get_competitor_news    : Recent competitor movement / news / FDA events.
  - summarize_article      : Deep-dive on one previously shown headline.
  - general_advisor        : Everything else, including definitional
                             questions about who our competitors are, what
                             Keytruda is, PD-1 / PD-L1 mechanism, market
                             access, and meta questions.

Rules for calling tools:
1. ALWAYS respond by calling one or more tools — never answer in plain
   prose. If the question is purely conversational or definitional, call
   `general_advisor`.
2. If the user asks several things in one turn (e.g. "top HCP in NJ AND
   recent Opdivo news AND how to market to them"), emit tool calls in the
   order a human would read the answers: opportunity → marketing → news.
3. Extract structured arguments whenever possible:
     - `state` as a two-letter USPS code.
     - `npi` only if the user quotes a 10-digit number.
     - `competitors` only for drugs the user explicitly named.
4. Only use `summarize_article` when the user clearly refers to a
   previously shown headline (cues: "tell me more", "summarize", "dig
   into", "deep dive", "that article", "headline #N", or pasting a title
   fragment). Pick the correct `headline_index` from the list the caller
   provides in the system message.
5. Do NOT classify questions as competitor-news just because a competitor
   name appears. News requires a recent-event cue ("news", "latest",
   "recent", "update", "announcement", "movement", "FDA", "approval",
   "readout").
6. Do NOT reorder or drop intents the user clearly asked about.
7. COUNTING vs LOOKUP disambiguation (important):
   - Questions that ask HOW MANY / HOW MUCH / AVERAGE / TYPICAL / WHAT
     SPECIALTIES / WHAT STATES etc. about the dataset are AGGREGATE /
     STATISTICAL questions. Route them to `general_advisor`, NOT to
     `get_hcp_opportunity`.
   - Only use `get_hcp_opportunity` when the user asks about ONE
     specific provider ("why is NPI ... a target", "who is the top HCP
     in NJ", "tell me about this doctor").
   - Examples:
       "How many providers are in our dataset?"   → general_advisor
       "What's a typical HCP profile?"            → general_advisor
       "How many states do we cover?"             → general_advisor
       "Why is NPI 1234567890 a priority?"        → get_hcp_opportunity
       "Top provider in NJ?"                      → get_hcp_opportunity
8. CRITICAL — tool argument formatting:
   - NEVER include a parameter with a value of `null`, `"null"`, `"None"`,
     empty string, or empty array. If you don't have a value, OMIT the
     key entirely from the JSON arguments object.
   - Example (GOOD):  {"competitors": ["opdivo"]}
   - Example (BAD):   {"competitors": ["opdivo"], "state": null}

You MUST respond with tool calls, not prose.
""".strip()


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def _format_cached_headlines_block(cached_headlines: list[dict] | None) -> str:
    """Render the currently cached news headlines with indices so the
    planner can target them by `headline_index`. Returns an empty string
    if there are no cached headlines."""
    if not cached_headlines:
        return ""
    lines = ["PREVIOUSLY SHOWN HEADLINES (use these indices for summarize_article):"]
    for i, item in enumerate(cached_headlines[:10]):
        title = (item.get("title") or "(untitled)").strip()
        source = (item.get("source") or "").strip()
        date = (item.get("date") or "").strip()
        meta = " · ".join([b for b in [source, date] if b])
        lines.append(f"  [{i}] {title}" + (f"  — {meta}" if meta else ""))
    return "\n".join(lines)


def _format_history_block(history: list[dict] | None, max_msgs: int = 4) -> str:
    """Compact summary of the last few turns so the planner has short-term
    memory. We only pass role + a truncated snippet so the prompt stays
    small and cheap.
    """
    if not history:
        return ""
    tail = history[-max_msgs:]
    lines = ["RECENT CONVERSATION (most recent last):"]
    for msg in tail:
        role = msg.get("role", "user")
        content = (msg.get("content") or "").strip().replace("\n", " ")
        if len(content) > 240:
            content = content[:240] + "..."
        lines.append(f"  - {role}: {content}")
    return "\n".join(lines)


def plan_actions(
    user_input: str,
    api_key: str,
    cached_headlines: list[dict] | None = None,
    history: list[dict] | None = None,
    model: str = "llama-3.1-8b-instant",
) -> list[dict] | None:
    """
    Ask the LLM to emit an ordered list of tool calls for this user turn.

    Returns:
      - list of dicts like {"name": str, "args": dict, "id": str} on success,
        possibly empty if the model chose to respond in prose.
      - None if the planner call itself failed (caller should fall back
        to the legacy `src.router.get_intents` path).
    """
    global LAST_ERROR
    LAST_ERROR = None
    try:
        client = Groq(api_key=api_key)

        # Build a single system message with the static planner prompt plus
        # any dynamic context (cached headlines, prior turns) the planner
        # may need for `summarize_article` or topic continuity.
        context_blocks = [PLANNER_SYSTEM_PROMPT]
        headlines_block = _format_cached_headlines_block(cached_headlines)
        if headlines_block:
            context_blocks.append(headlines_block)
        history_block = _format_history_block(history)
        if history_block:
            context_blocks.append(history_block)
        system_content = "\n\n".join(context_blocks)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_input},
            ],
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
            max_tokens=512,
        )

        choice = response.choices[0]
        tool_calls = getattr(choice.message, "tool_calls", None) or []

        plan: list[dict] = []
        for tc in tool_calls:
            # Groq returns OpenAI-compatible objects: tc.function.name / .arguments (JSON string).
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            name = getattr(fn, "name", None)
            if name not in VALID_TOOL_NAMES:
                continue
            raw_args = getattr(fn, "arguments", "") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except (json.JSONDecodeError, TypeError):
                args = {}
            if not isinstance(args, dict):
                args = {}
            # Smaller models (e.g. llama-3.1-8b) sometimes emit explicit
            # `null` / empty values for optional params instead of omitting
            # them. Strip those so downstream handlers see clean args.
            args = {
                k: v for k, v in args.items()
                if v is not None and v != "" and v != []
            }
            plan.append({
                "name": name,
                "args": args,
                "id": getattr(tc, "id", "") or "",
            })

        # If the model answered in plain prose (no tool_calls), we surface
        # that so the caller can treat it as a general_advisor response.
        if not plan:
            prose = (choice.message.content or "").strip()
            if prose:
                return [{
                    "name": "general_advisor",
                    "args": {"question": user_input, "_prose": prose},
                    "id": "",
                }]

        return plan

    except Exception as e:  # pragma: no cover - network / library failures
        # Special-case: `tool_use_failed` means Groq's server-side tool-call
        # validator rejected the model's output (typically because a smaller
        # model emitted `null` for an optional param). The error payload
        # contains the raw tool call the model wanted to make — we can
        # parse it, strip the offending nulls, and recover instead of
        # failing the whole turn.
        recovered = _recover_from_tool_use_failed(e, user_input)
        if recovered is not None:
            print(
                f"Planner recovered from tool_use_failed: "
                f"{[p.get('name') for p in recovered]}"
            )
            return recovered

        tb = traceback.format_exc()
        print(f"Planner Error (plan_actions): {type(e).__name__}: {e}")
        print(tb)
        LAST_ERROR = {
            "type": type(e).__name__,
            "message": str(e)[:500],
            "traceback": tb[-2000:],
        }
        return None


def _recover_from_tool_use_failed(
    exc: Exception, user_input: str
) -> list[dict] | None:
    """Try to salvage a plan from a Groq `tool_use_failed` BadRequestError.

    The error body contains `failed_generation`, which is the raw string the
    model emitted, e.g.:
        <function=get_competitor_news>{"competitors": ["opdivo"], "state": null}</function>

    We regex-parse the function name and JSON args, drop any None/empty
    values, and return a plan the caller can dispatch. Returns None if
    this isn't a recoverable error.
    """
    import re

    msg = str(exc)
    if "tool_use_failed" not in msg and "did not match schema" not in msg:
        return None

    match = re.search(
        r"<function=([a-zA-Z_][a-zA-Z0-9_]*)>\s*(\{.*?\})\s*</function>",
        msg,
        flags=re.DOTALL,
    )
    if not match:
        return None

    name = match.group(1)
    raw_args = match.group(2)
    if name not in VALID_TOOL_NAMES:
        return None

    try:
        args = json.loads(raw_args)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(args, dict):
        return None

    args = {
        k: v for k, v in args.items()
        if v is not None and v != "" and v != [] and v != "null"
    }
    return [{"name": name, "args": args, "id": ""}]
