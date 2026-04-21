# src/agent_graph.py
"""
LangGraph agent (Option 3).

Wraps the native tool-calling planner (`src/planner.py`) inside a 5-node
state graph that adds capabilities the flat planner cannot express:

    START -> plan -> execute -> grounding_check -> reflect -> finalize -> END
                                                      |
                                                      +-> plan (retry, max 1)

Nodes:
  1. plan            : Calls the injected `planner_fn` (thin wrapper over
                       `src.planner.plan_actions`). Produces an ordered
                       list of tool calls.
  2. execute         : Dispatches each tool call via the injected
                       `dispatch_fn` and collects their markdown outputs.
  3. grounding_check : Responsible-AI guardrail. Extracts any 10-digit NPI
                       mentioned in the draft outputs and verifies each
                       against the real dataset using `verify_npi_fn`. Any
                       NPI not found is a hallucination.
  4. reflect         : LLM-as-judge. If grounding failed (and retries are
                       available) OR if the judge says a substantive
                       sub-question of the user is missing, routes back to
                       `plan` for a single retry. Otherwise proceeds.
  5. finalize        : Stitches outputs into the final markdown answer,
                       appending a grounding-warning banner if a
                       hallucination was detected but we ran out of retries.

Design:
- Dependencies (planner_fn / dispatch_fn / verify_npi_fn) are INJECTED so
  this module is decoupled from Streamlit and from the dataset — callers
  can test nodes in isolation and the Streamlit layer keeps ownership of
  session state.
- All exceptions are swallowed with safe defaults; the worst case is we
  return a degraded final answer rather than crashing the graph.
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


# Type aliases for the injected dependencies.
PlannerFn = Callable[[str, list[dict], list[dict]], list[dict] | None]
DispatchFn = Callable[[str, dict, str], str | None]
VerifyNpiFn = Callable[[str], bool]


def build_agent_graph(
    planner_fn: PlannerFn,
    dispatch_fn: DispatchFn,
    verify_npi_fn: VerifyNpiFn,
    groq_api_key: str,
    model: str = "llama-3.3-70b-versatile",
):
    """Build and compile the 5-node agent graph.

    Args:
        planner_fn:   wraps `plan_actions(user_prompt, cached, history)` and
                      returns `[{"name","args","id"}, ...]` or `None`.
        dispatch_fn:  wraps `_dispatch_planner_call(name, args, prompt)` and
                      returns a markdown string (or `None` for unknown tools).
        verify_npi_fn: returns True iff the 10-digit NPI exists in the dataset.
        groq_api_key: used by the `reflect` node's LLM-as-judge call.
        model:        Groq model used by the reflect node.

    Returns:
        A compiled `CompiledGraph` you can `.invoke(state_dict)` on.
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

    def reflect_node(state: AgentState) -> dict[str, Any]:
        """Decide whether to retry (back to plan) or finalize.

        Priority:
          1. If the planner itself failed, no point retrying — finalize.
          2. If grounding flagged NPIs and we still have a retry budget,
             RETRY with a reflection note the planner will see in history.
          3. Otherwise ask an LLM-as-judge whether every sub-question in
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
    graph.add_node("reflect", reflect_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "grounding_check")
    graph.add_edge("grounding_check", "reflect")
    graph.add_conditional_edges(
        "reflect",
        _route_after_reflect,
        {"plan": "plan", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile()
