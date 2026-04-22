# Merck Keytruda — Provider Targeting Strategy AI

An agentic Streamlit application that pairs an internal HCP-propensity model
with an LLM-driven advisory layer to answer four classes of commercial
questions in one chat surface:

1. **HCP opportunity targeting** — which provider to prioritize in a given
   state, grounded in a propensity-score scorecard.
2. **Marketing strategy (Next-Best-Action)** — per-HCP channel / timing /
   message recommendations.
3. **Competitor news + share context** — live DuckDuckGo news search for
   Opdivo / Tecentriq / Imfinzi / Libtayo, fused with internal billing-mix
   share from the CSV.
4. **General Keytruda / oncology / commercial-strategy Q&A.**

The app is deployed on Streamlit Community Cloud and uses Groq
(`llama-3.1-8b-instant`) as the primary LLM provider.

---

## Start here

**[→ docs/CAPABILITIES.md](docs/CAPABILITIES.md)** — a 2-minute summary of
the architecture / patterns demonstrated, the current capability surface,
and prioritized future work. Start there; everything else is a deep dive.

---

## What's interesting about this repo

This is *not* an "LLM-wrapped-around-a-CSV" demo. It is a small but real
demonstration of production-leaning GenAI patterns:

- **Three-layer routing stack** with graceful fallback — a LangGraph agent
  on top, a Groq tool-calling planner in the middle, and a deterministic
  intent classifier at the bottom. Every layer can be toggled independently
  via feature flags so the team can ship incrementally and roll back
  without a redeploy.
- **Two Responsible-AI guardrail nodes** inside the LangGraph:
  a deterministic grounding check that verifies every cited NPI exists in
  the real dataset, and an LLM-as-judge competitor-claim check that flags
  competitor drug facts not supported by cached news evidence.
- **Bounded reflection loop** — at most one replan per user turn on
  guardrail failure, so worst-case latency is deterministic.
- **Clean separation of concerns** — routing (`src/planner.py`,
  `src/agent_graph.py`) is side-effect-free. Streamlit only handles UI
  and session state. Every node in the graph takes its dependencies by
  injection so each is individually testable.

---

## Documentation map

| Document | What's in it |
|---|---|
| **[docs/CAPABILITIES.md](docs/CAPABILITIES.md)** | **Start here.** 2-minute scan — architecture demonstrated, current capabilities, future work. |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, three-layer routing model, data flow, key decisions and tradeoffs |
| [docs/ROUTING.md](docs/ROUTING.md) | Deep dive on each routing layer and the v1 → v2 → v3 evolution story |
| [docs/GUARDRAILS.md](docs/GUARDRAILS.md) | Responsible-AI layer: grounding check, competitor-claim check, design principles, tuning case study |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Prioritized future work with T-shirt sizing and explicit non-goals |
| [docs/agent_graph.md](docs/agent_graph.md) | Auto-generated Mermaid topology of the LangGraph agent |

---

## Quick start (local)

```bash
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` with at minimum:

```toml
GROQ_API_KEY = "your-groq-key"

# Optional routing flags (defaults shown)
USE_LANGGRAPH  = true    # top-of-stack agentic path
USE_PLANNER    = true    # middle tool-calling path
DEBUG_ROUTING  = false   # surface tracebacks in the UI; keep off in prod
```

Then:

```bash
streamlit run streamlit_app.py
```

## Quick start (Streamlit Community Cloud)

1. Fork / push the repo.
2. On Streamlit Cloud, point the app at `streamlit_app.py`.
3. Add the secrets above in the app's **Secrets** tab.
4. Deploy. Cache-clear + hard-refresh after any secrets change.

---

## Regenerate the agent-graph diagram

The LangGraph topology is auto-documented. After changing nodes or edges in
`src/agent_graph.py`, run:

```bash
python scripts/render_graph.py
```

This overwrites `docs/agent_graph.md` with a fresh Mermaid diagram.

---

## License

See [LICENSE](LICENSE).
