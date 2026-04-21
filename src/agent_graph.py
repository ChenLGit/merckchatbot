# src/agent_graph.py
"""
LangGraph agent (Option 3).

Wraps the native tool-calling planner (`src/planner.py`) inside a 6-node
state graph that adds two responsible-AI guardrails on top of a reflection
loop:

    START -> plan -> execute -> grounding_check -> competitor_claim_check
                                                        |
                                                        v
                                                     reflect -> finalize -> END
                                                        |
                                                        +-> plan (retry, max 1)

Nodes:
  1. plan                    : Calls the injected `planner_fn` (thin
                               wrapper over `src.planner.plan_actions`).
                               Produces an ordered list of tool calls.
  2. execute                 : Dispatches each tool call via the injected
                               `dispatch_fn` and collects markdown outputs.
  3. grounding_check         : Responsible-AI guardrail #1. Extracts any
                               10-digit NPI mentioned in the draft and
                               verifies it exists in the targeting dataset
                               via `verify_npi_fn`. Any NPI not found is a
                               hallucination.
  4. competitor_claim_check  : Responsible-AI guardrail #2. Uses
                               LLM-as-judge to fact-check any SPECIFIC,
                               VERIFIABLE claim about a competitor drug
                               (Opdivo, Tecentriq, Imfinzi, Libtayo)
                               against the cached news body provided by
                               `news_snapshot_fn`. Generic/qualitative
                               statements are ignored.
  5. reflect                 : LLM-as-judge on coverage. Routes back to
                               `plan` for at most one retry if (a)
                               grounding flagged hallucinated NPIs, (b)
                               claim check flagged unsupported competitor
                               claims, or (c) a substantive sub-question
                               was completely missed. Otherwise proceeds.
  6. finalize                : Stitches outputs into final markdown,
                               appending warning banners for any guardrail
                               issues that survived the retry budget.

Design:
- Dependencies (planner_fn / dispatch_fn / verify_npi_fn / news_snapshot_fn)
  are INJECTED so this module is decoupled from Streamlit and from the
  dataset. Nodes are individually testable.
- All LLM / lookup calls swallow exceptions and fail OPEN (treat as OK)
  so a transient judge failure never blocks a legitimate answer.
- The retry budget (`MAX_RETRIES = 1`) keeps worst-case latency bounded.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Literal, TypedDict

from groq import Groq
from langgraph.graph import END, START, StateGraph


# -----------------------------------------------------------------------------
# Typed state
# -----------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    """Single source of truth for the graph. Every node reads/writes this."""

    # ---- Inputs (seeded before invoke) ----
    user_prompt: str
    cached_headlines: list[dict]
    history: list[dict]

    # ---- plan node output ----
    plan: list[dict]
    planner_failed: bool

    # ---- execute node output ----
    outputs: list[str]

    # ---- grounding_check node output ----
    grounding: Literal["OK", "HALLUCINATION"]
    grounding_note: str
    flagged_npis: list[str]

    # ---- competitor_claim_check node output ----
    competitor_claim: Literal["OK", "UNSUPPORTED"]
    competitor_claim_note: str

    # ---- reflect node output ----
    reflection: Literal["OK", "RETRY"]
    reflection_note: str
    retry_count: int

    # ---- finalize node output ----
    final_answer: str


# At most one re-plan per user turn (to keep latency bounded).
MAX_RETRIES = 1

# Standalone 10-digit NPI in free text.
_NPI_PATTERN = re.compile(r"(?<!\d)(\d{10})(?!\d)")


# Tokens that, when present in a draft output, indicate the assistant
# referenced a direct competitor drug or manufacturer. Used as a cheap
# short-circuit so the claim-check LLM call only fires when relevant.
_COMPETITOR_TOKENS: tuple[str, ...] = (
    "opdivo", "nivolumab",
    "tecentriq", "atezolizumab",
    "imfinzi", "durvalumab",
    "libtayo", "cemiplimab",
    "bristol-myers", "bristol myers", "bms",
    "genentech", "roche",
    "astrazeneca", "astra zeneca",
    "regeneron", "sanofi",
)


# Type aliases for the injected dependencies.
PlannerFn = Callable[[str, list[dict], list[dict]], list[dict] | None]
DispatchFn = Callable[[str, dict, str], str | None]
VerifyNpiFn = Callable[[str], bool]
# Returns the most-recent cached news items (list of dicts with title /
# body / source / date / url keys, as produced by `src/news_engine.py`).
# The claim-check node uses this as its only ground truth for any specific
# recent-event claim about a competitor drug.
NewsSnapshotFn = Callable[[], list[dict]]


def build_agent_graph(
    planner_fn: PlannerFn,
    dispatch_fn: DispatchFn,
    verify_npi_fn: VerifyNpiFn,
    groq_api_key: str,
    model: str = "llama-3.3-70b-versatile",
    news_snapshot_fn: NewsSnapshotFn | None = None,
):
    """Build and compile the 6-node agent graph.

    Args:
        planner_fn:       wraps `plan_actions(user_prompt, cached, history)`
                          and returns `[{"name","args","id"}, ...]` or `None`.
        dispatch_fn:      wraps `_dispatch_planner_call(name, args, prompt)`
                          and returns a markdown string (or `None` for
                          unknown tools).
        verify_npi_fn:    returns True iff the 10-digit NPI exists in the
                          targeting dataset.
        groq_api_key:     used by reflect and competitor_claim_check nodes
                          for their LLM-as-judge calls.
        model:            Groq model used by the judge calls.
        news_snapshot_fn: returns the most-recent cached news items (list
                          of dicts). Used by competitor_claim_check as its
                          evidence base. If omitted, the claim-check node
                          fails OPEN (can't verify, so passes).

    Returns:
        A compiled `CompiledGraph` you can `.invoke(state_dict)` or
        `.stream(state_dict, stream_mode="updates")` on.
    """

    # -----------------------------------------------------------------
    # Nodes
    # -----------------------------------------------------------------
    def plan_node(state: AgentState) -> dict[str, Any]:
        plan = planner_fn(
            state.get("user_prompt", ""),
            state.get("cached_headlines") or [],
            state.get("history") or [],
        )
        if plan is None:
            # Hard planner failure — keep flowing but mark it so the
            # finalize node can surface a graceful message.
            return {"plan": [], "planner_failed": True}
        if not plan:
            plan = [{
                "name": "general_advisor",
                "args": {"question": state.get("user_prompt", "")},
                "id": "",
            }]
        return {"plan": plan, "planner_failed": False}

    def execute_node(state: AgentState) -> dict[str, Any]:
        outputs: list[str] = []
        for call in state.get("plan") or []:
            try:
                out = dispatch_fn(
                    call.get("name") or "",
                    call.get("args") or {},
                    state.get("user_prompt", ""),
                )
            except Exception as e:  # noqa: BLE001
                out = f"_(Tool `{call.get('name')}` failed: {str(e)[:80]})_"
            if out:
                outputs.append(out)
        return {"outputs": outputs}

    def grounding_check_node(state: AgentState) -> dict[str, Any]:
        """Verify any 10-digit NPI that appears in the draft outputs is
        a real row in our targeting table. Flag hallucinations."""
        draft = "\n\n".join(state.get("outputs") or [])
        mentioned = sorted(set(_NPI_PATTERN.findall(draft)))
        flagged: list[str] = []
        for npi in mentioned:
            try:
                ok = bool(verify_npi_fn(npi))
            except Exception:
                # Fail OPEN — don't block a legit answer on a lookup error.
                ok = True
            if not ok:
                flagged.append(npi)
        if flagged:
            return {
                "grounding": "HALLUCINATION",
                "grounding_note": f"NPIs not in dataset: {', '.join(flagged)}",
                "flagged_npis": flagged,
            }
        return {"grounding": "OK", "grounding_note": "", "flagged_npis": []}

    def competitor_claim_check_node(state: AgentState) -> dict[str, Any]:
        """Fact-check specific competitor-drug claims against cached news.

        Short-circuits when no competitor name appears in the draft, so
        the LLM-as-judge call only fires when relevant. Always fails OPEN
        on any internal error — a broken checker must never block a
        legitimate answer.
        """
        draft = "\n\n".join(state.get("outputs") or [])
        low = draft.lower()
        if not any(tok in low for tok in _COMPETITOR_TOKENS):
            return {"competitor_claim": "OK", "competitor_claim_note": ""}

        news_items: list[dict] = []
        if news_snapshot_fn is not None:
            try:
                news_items = list(news_snapshot_fn() or [])
            except Exception:  # noqa: BLE001
                news_items = []

        if not news_items:
            # Competitor mentioned but no news evidence available to verify
            # against. Fail OPEN and let reflect / finalize proceed.
            return {
                "competitor_claim": "OK",
                "competitor_claim_note": "no cached news to verify against",
            }

        evidence_lines = []
        for i, it in enumerate(news_items[:8]):
            title = (it.get("title") or "").strip()
            body = (it.get("body") or "").strip()
            source = (it.get("source") or "").strip()
            date = (it.get("date") or "").strip()
            meta = " · ".join([b for b in [source, date] if b])
            evidence_lines.append(
                f"[{i}] {title}" + (f"  ({meta})" if meta else "")
                + (f"\n    {body[:500]}" if body else "")
            )
        evidence = "\n".join(evidence_lines)

        verdict, note = "OK", ""
        try:
            client = Groq(api_key=groq_api_key)
            judge_prompt = f"""
You are fact-checking a Merck Keytruda brand assistant's DRAFT answer for
claims about competitor drugs (Opdivo / nivolumab, Tecentriq / atezolizumab,
Imfinzi / durvalumab, Libtayo / cemiplimab) or their manufacturers
(BMS, Roche/Genentech, AstraZeneca, Regeneron/Sanofi).

DRAFT ANSWER:
{draft}

AVAILABLE NEWS EVIDENCE (the ONLY ground truth for recent-event claims):
{evidence}

TASK:
Identify every SPECIFIC, VERIFIABLE claim about a competitor drug or
maker in the draft — e.g. "FDA approved X on DATE", "Opdivo showed a 20%
response rate", "BMS announced Y partnership". For each, decide whether
the evidence above supports it.

IGNORE:
- Generic / qualitative statements ("Opdivo is a PD-1 inhibitor",
  "Opdivo remains a key competitor", "the IO space is crowded").
- Keytruda-only statements (we only fact-check competitor claims here).
- Market-share framing that references internal billing mix (that is
  sourced from our own CSV, not news).

Respond with JSON ONLY, matching this schema:

{{"verdict": "OK" | "UNSUPPORTED",
  "note": "<if UNSUPPORTED, list the offending claim(s), <= 200 chars>"}}

- "OK" if every specific competitor claim is supported by the evidence,
  OR if the draft only contains generic/qualitative statements.
- "UNSUPPORTED" if even one specific claim is not in the evidence.
""".strip()

            res = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(res.choices[0].message.content or "{}")
            verdict = str(parsed.get("verdict", "OK")).upper()
            note = str(parsed.get("note", ""))[:240]
        except Exception as e:  # noqa: BLE001
            verdict, note = "OK", f"claim_check_error: {str(e)[:80]}"

        if verdict == "UNSUPPORTED":
            return {
                "competitor_claim": "UNSUPPORTED",
                "competitor_claim_note": note,
            }
        return {"competitor_claim": "OK", "competitor_claim_note": note}

    def reflect_node(state: AgentState) -> dict[str, Any]:
        """Decide whether to retry (back to plan) or finalize.

        Priority:
          1. If the planner itself failed, no point retrying — finalize.
          2. If grounding flagged hallucinated NPIs and we still have a
             retry budget, RETRY with a correction note.
          3. If claim-check flagged unsupported competitor claims and we
             still have a retry budget, RETRY with a correction note.
          4. Otherwise ask an LLM-as-judge whether every sub-question in
             the user's prompt was addressed. Only RETRY on a clear miss
             and only while retry budget remains.
        """
        retry_count = int(state.get("retry_count") or 0)

        if state.get("planner_failed"):
            return {"reflection": "OK", "reflection_note": "planner_failed"}

        if (
            state.get("grounding") == "HALLUCINATION"
            and retry_count < MAX_RETRIES
        ):
            return {
                "reflection": "RETRY",
                "reflection_note": (
                    "Grounding failure. " + (state.get("grounding_note") or "")
                    + " Re-plan and AVOID mentioning unverified NPIs."
                ),
                "retry_count": retry_count + 1,
            }

        if (
            state.get("competitor_claim") == "UNSUPPORTED"
            and retry_count < MAX_RETRIES
        ):
            return {
                "reflection": "RETRY",
                "reflection_note": (
                    "Unsupported competitor claim(s). "
                    + (state.get("competitor_claim_note") or "")
                    + " Re-plan and only state competitor facts that are in "
                    "the cached news evidence."
                ),
                "retry_count": retry_count + 1,
            }

        # LLM-as-judge. If we're already out of retries, skip the call.
        if retry_count >= MAX_RETRIES:
            return {"reflection": "OK", "reflection_note": ""}

        draft = "\n\n".join(state.get("outputs") or [])
        if not draft.strip():
            return {"reflection": "OK", "reflection_note": "empty_draft"}

        verdict, note = "OK", ""
        try:
            client = Groq(api_key=groq_api_key)
            judge_prompt = f"""
You are a QA reviewer for a Merck Keytruda strategy assistant.

USER QUESTION:
{state.get('user_prompt', '')}

ASSISTANT DRAFT ANSWER:
{draft}

TASK:
Decide if the draft fully addresses every substantive sub-question the
user asked. Respond with JSON only, no preamble, matching this schema:

{{"verdict": "OK" | "RETRY", "note": "<short reason, <=120 chars>"}}

Rules:
- "OK" if every sub-question is addressed (even briefly).
- "RETRY" only if a substantive sub-question is COMPLETELY missing.
- Do NOT retry just to request more detail on a topic already covered.
- Do NOT retry for stylistic reasons.
""".strip()

            res = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0,
                max_tokens=120,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(res.choices[0].message.content or "{}")
            verdict = str(parsed.get("verdict", "OK")).upper()
            note = str(parsed.get("note", ""))[:200]
        except Exception as e:  # noqa: BLE001
            # Fail CLOSED on the retry — never loop on a broken reflect.
            verdict, note = "OK", f"reflect_error: {str(e)[:80]}"

        if verdict == "RETRY" and retry_count < MAX_RETRIES:
            return {
                "reflection": "RETRY",
                "reflection_note": note,
                "retry_count": retry_count + 1,
            }
        return {"reflection": "OK", "reflection_note": note}

    def finalize_node(state: AgentState) -> dict[str, Any]:
        """Stitch outputs into the final markdown answer."""
        if state.get("planner_failed"):
            return {
                "final_answer": (
                    "_(Planner unavailable — please try rephrasing your "
                    "question.)_"
                )
            }

        parts: list[str] = list(state.get("outputs") or [])
        if state.get("grounding") == "HALLUCINATION":
            flagged = state.get("flagged_npis") or []
            parts.append(
                "> ⚠️ **Grounding check:** the draft above referenced NPIs "
                f"that are not in the targeting dataset "
                f"({', '.join(flagged)}). Treat those references with caution."
            )
        if state.get("competitor_claim") == "UNSUPPORTED":
            claim_note = (state.get("competitor_claim_note") or "").strip()
            parts.append(
                "> ⚠️ **Competitor claim check:** one or more specific "
                "claims about a competitor drug could not be verified "
                "against the cached news evidence."
                + (f" _Details: {claim_note}_" if claim_note else "")
            )
        if not parts:
            return {"final_answer": "_(No answer produced.)_"}
        return {"final_answer": "\n\n---\n\n".join(parts)}

    # -----------------------------------------------------------------
    # Routing
    # -----------------------------------------------------------------
    def _route_after_reflect(state: AgentState) -> str:
        if state.get("reflection") == "RETRY":
            return "plan"
        return "finalize"

    # -----------------------------------------------------------------
    # Wire the graph
    # -----------------------------------------------------------------
    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("grounding_check", grounding_check_node)
    graph.add_node("competitor_claim_check", competitor_claim_check_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "grounding_check")
    graph.add_edge("grounding_check", "competitor_claim_check")
    graph.add_edge("competitor_claim_check", "reflect")
    graph.add_conditional_edges(
        "reflect",
        _route_after_reflect,
        {"plan": "plan", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile()
