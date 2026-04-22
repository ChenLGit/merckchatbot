# =============================================================================
# Merck Keytruda Streamlit Application
# =============================================================================

# -------------------
# Imports & Path Setup
# -------------------
import sys
import os
import streamlit as st
import pandas as pd
from groq import Groq

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from src.router import get_intent, get_intents
from src.planner import plan_actions
from src.visualizations import plot_executive_map
from src.rag_engine import get_hcp_scorecard, extract_npi
from src.news_engine import get_competitor_brief

try:
    from src.prompts import SYSTEM_PERSONAS, build_system_prompt
except ImportError:
    SYSTEM_PERSONAS = {}

    def build_system_prompt(role_key):  # type: ignore[no-redef]
        return SYSTEM_PERSONAS.get(role_key, "You are a Merck Keytruda brand advisor.")

# ---------------------------------
# Helpers: Format Outputs
# ---------------------------------
import re as _re

# Emoji/markers the marketing persona is supposed to start each bullet with.
_MARKETING_SECTION_MARKERS = (
    "🎯",  # Primary Recommendation
    "🛠️",  # Tactical Channel
    "🛠",   # Tactical Channel (no variation selector)
    "⏱️",  # Timing
    "⏱",   # Timing (no variation selector)
)

def _normalize_marketing_text(text: str) -> str:
    """Force each '🎯 ... 🛠️ ... ⏱️ ...' section onto its own line with no
    blank line in between. Uses markdown hard line breaks (two trailing
    spaces + newline) so Streamlit renders them tight instead of as a
    'loose' bulleted list with paragraph spacing.
    """
    if not text:
        return text
    s = str(text).strip()

    # Split the text so every marker starts a new line, even if the LLM
    # ran them together in one paragraph.
    for marker in _MARKETING_SECTION_MARKERS:
        s = _re.sub(rf"(?<!\n)\s*(?={_re.escape(marker)})", "\n", s)

    cleaned = []
    for raw_line in s.split("\n"):
        ln = raw_line.strip()
        if not ln:
            continue
        # Strip any leading bullet/number markers the LLM may have added.
        ln = _re.sub(r"^(?:[-*]|\d+[.)])\s+", "", ln)
        cleaned.append(ln)

    # '  \n' == markdown hard line break -> new line, no blank gap.
    return "  \n".join(cleaned)


def _lookup_label(scorecard):
    """Small tag that tells the user whether this was a specific NPI lookup
    or the top-ranked provider for the inferred filter."""
    if scorecard.get("matched_by_npi"):
        return f"_Lookup: requested NPI {scorecard.get('requested_npi')}_"
    return "_Lookup: top propensity for inferred filter_"

def _format_opportunity_markdown(scorecard, insight_text=None):
    lines = [
        f"### 🎯 NPI: {scorecard['npi']}",
        _lookup_label(scorecard),
        f"**Location:** {scorecard.get('city', 'N/A')}, {scorecard['state']} | **Type:** {scorecard['type']}",
        "**Opportunity Scorecard:**",
        f"- **Propensity Score:** {scorecard['score']:.1%}",
        f"- **Avg Medicare Payment per Person:** ${scorecard['payment']:,.2f}",
        f"- **Top Model Drivers:** {scorecard['drivers']}",
    ]
    md = "\n\n".join(lines)
    if insight_text:
        md += f"\n\n💡 **AI Insight:** {insight_text}"
    return md

def _short_date(date_str):
    """Turn '2026-03-31T00:00:00+00:00' into '2026-03-31'. Keep anything
    that doesn't match the ISO pattern untouched."""
    if not date_str:
        return ""
    s = str(date_str).strip()
    m = _re.match(r"(\d{4}-\d{2}-\d{2})", s)
    return m.group(1) if m else s


# ---------------------------------
# News article summarization helpers
# ---------------------------------
_SUMMARY_TRIGGER_WORDS = (
    "summarize", "summary", "summarise", "tl;dr", "tldr",
    "tell me more", "more about", "details on", "details about",
    "read more", "explain this news", "what does it say", "what does this say",
    "dig into", "elaborate on", "deep dive", "full story",
)


def _tokenize_for_match(text):
    """Lowercase + keep only alphanumeric tokens of length >= 2."""
    if not text:
        return []
    return [t for t in _re.findall(r"[a-z0-9]+", str(text).lower()) if len(t) >= 2]


def _find_news_item_for_query(query, items):
    """Best-effort match: does `query` look like it's referring to one of the
    cached news headlines? Returns the matching news dict or None.

    We match if any of:
      - >= 4 consecutive tokens from a headline appear in the user query, or
      - >= 60% of the headline's content tokens appear in the query.
    """
    if not items or not query:
        return None

    q_tokens = _tokenize_for_match(query)
    if not q_tokens:
        return None
    q_joined = " ".join(q_tokens)

    best = None
    best_score = 0.0
    _STOP = {
        "the", "and", "for", "with", "from", "that", "this",
        "will", "has", "have", "a", "an", "of", "to", "in", "on",
        "at", "by", "as", "is", "are", "was", "were", "be", "it",
        "its", "or", "but", "not", "vs", "more", "less",
    }
    for item in items:
        title_tokens = [t for t in _tokenize_for_match(item.get("title")) if t not in _STOP]
        if not title_tokens:
            continue

        # Test 1: contiguous 4-gram overlap in the original query.
        contiguous_hit = False
        for i in range(len(title_tokens) - 3):
            gram = " ".join(title_tokens[i:i + 4])
            if gram in q_joined:
                contiguous_hit = True
                break

        # Test 2: content-word overlap ratio.
        overlap = len(set(title_tokens) & set(q_tokens))
        ratio = overlap / max(len(title_tokens), 1)

        score = ratio + (0.5 if contiguous_hit else 0.0)
        if score > best_score and (contiguous_hit or ratio >= 0.6):
            best_score = score
            best = item
    return best


def _looks_like_summary_request(query):
    """True if the query contains an explicit 'summarize/tell me more' cue."""
    q = (query or "").lower()
    return any(trigger in q for trigger in _SUMMARY_TRIGGER_WORDS)


def _fetch_article_text(url, max_chars=6000):
    """Best-effort fetch of an article and a rough plain-text extract. Returns
    '' on any failure (network blocked, paywall, non-HTML, missing lib)."""
    if not url:
        return ""
    try:
        import requests  # transitive via streamlit/groq
    except ImportError:
        return ""
    try:
        resp = requests.get(
            url,
            timeout=8,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; MerckKeytrudaBot/1.0; "
                    "+https://merck.com)"
                )
            },
        )
        if resp.status_code >= 400 or not resp.text:
            return ""
        html = resp.text
    except Exception:
        return ""

    # Drop scripts and styles first, then strip all remaining HTML tags.
    html = _re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = _re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    html = _re.sub(r"(?is)<!--.*?-->", " ", html)
    text = _re.sub(r"(?s)<[^>]+>", " ", html)
    # Collapse whitespace and decode a few common entities.
    text = (
        text.replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&quot;", '"')
            .replace("&#39;", "'")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
    )
    text = _re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def _format_news_summary_markdown(item, summary_text):
    title = item.get("title") or "(untitled)"
    source = item.get("source") or ""
    date = _short_date(item.get("date") or "")
    url = item.get("url") or ""
    meta = " · ".join([b for b in [source, date] if b])
    header_title = f"[{title}]({url})" if url else title
    lines = [
        "### 📰 Article Summary",
        f"##### {header_title}",
    ]
    if meta:
        lines.append(f"_{meta}_")
    if summary_text:
        lines.append(summary_text.strip())
    return "\n\n".join(lines)


def _format_news_markdown(brief, recommendation_text=None):
    """Render the competitor-news view. Structure:
       1. Title
       2. AI-synthesized "What happened / Competitor impact" commentary
       3. Recent headlines at the END
    Internal billing mix and anchor-HCP details are intentionally hidden so
    pure news questions don't get buried under HCP-targeting noise.
    """
    state = brief.get("state")
    scope_label = f"State: {state}" if state else "Scope: National"
    lines = [
        f"### 📰 Competitor Intelligence\n##### {scope_label}",
    ]

    if recommendation_text:
        lines.append(recommendation_text.strip())

    news = brief.get("news")
    if news is None:
        lines.append("_Live news search unavailable (package missing or network blocked)._")
    elif not news:
        lines.append("_No recent articles returned for the inferred competitors._")
    else:
        lines.append("**📰 Recent headlines:**")
        headline_bullets = []
        for item in news[:8]:
            title = item.get("title") or "(untitled)"
            source = item.get("source") or ""
            date = _short_date(item.get("date") or "")
            url = item.get("url") or ""
            meta_bits = " · ".join([b for b in [source, date] if b])
            if url:
                headline_bullets.append(f"- [{title}]({url}) — {meta_bits}")
            else:
                headline_bullets.append(f"- {title} — {meta_bits}")
        lines.append("\n".join(headline_bullets))
        lines.append(
            "_Want a deeper read? Paste a headline (or type "
            "`summarize <headline>`) and I'll summarize the article for you._"
        )

    return "\n\n".join(lines)


def _format_marketing_markdown(scorecard, strategy_text=None):
    """Renders the marketing strategy view for the same provider the
    opportunity view would surface (or the exact NPI the user asked for)."""
    try:
        digital_score = float(scorecard.get("digital_score", 0) or 0)
    except (TypeError, ValueError):
        digital_score = 0.0
    last_engagement = scorecard.get("last_engagement", 0)
    try:
        last_engagement = int(float(last_engagement))
    except (TypeError, ValueError):
        last_engagement = 0

    lines = [
        f"### 📈 Marketing Strategy\n##### NPI {scorecard['npi']}",
        _lookup_label(scorecard),
        f"**Location:** {scorecard.get('city', 'N/A')}, {scorecard['state']} | **Propensity:** {scorecard['score']:.1%}",
        "**Engagement Metrics:**",
        f"- **Digital Adoption Score:** {digital_score:.2f}",
        f"- **Last Engagement:** {last_engagement} days ago",
        f"- **Preferred Channel:** {scorecard.get('channel', 'Unknown')}",
    ]
    md = "\n\n".join(lines)
    if strategy_text:
        nba = _normalize_marketing_text(strategy_text)
        md += "\n\n**🎯 Next Best Action**  \n" + nba
    return md

# ==========================================================================
# 1. SETUP & DATA LOADING SECTION
# ==========================================================================
st.set_page_config(page_title="Merck Data Science Hub", layout="wide")

@st.cache_data
def load_data():
    data_path = os.path.join(current_dir, 'data', 'raw', 'MerckAI_table.csv')
    if not os.path.exists(data_path):
        st.error(f"Data file not found at: {data_path}")
        st.stop()
    df = pd.read_csv(data_path)
    shap_cols = [c for c in df.columns if c.startswith('SHAP_')]
    for col in shap_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace("'", ""), errors='coerce').fillna(0)
    return df

df = load_data()


@st.cache_data(ttl=600, show_spinner=False)
def _cached_competitor_brief(prompt_text: str):
    """Cache competitor briefs for 10 minutes per prompt so Streamlit reruns
    don't re-hit the network. The df is captured by reference from the outer
    scope but doesn't change during a session."""
    return get_competitor_brief(prompt_text, df)

# ==========================================================================
# 2. MAIN UI LAYOUT
# ==========================================================================
main_col, chat_col = st.columns([1.5, 1], gap="medium")

with main_col:
    header_left, header_right = st.columns([3.5, 1], vertical_alignment="top")
    with header_left:
        st.markdown("""
            <div style="text-align: left;">
                <div style="font-family: sans-serif; font-size: 3.8rem; font-weight: bold; color: #00857c; line-height: 1.1; margin-bottom: 8px;">
                    Merck Keytruda
                </div>
                <div style="font-family: sans-serif; font-size: 1.8rem; font-weight: normal; color: #555;">
                    Provider Targeting Strategy AI Application
                </div>
            </div>
        """, unsafe_allow_html=True)
    with header_right:
        st.markdown("""
            <div style="background-color: #f0f2f6; padding: 12px 18px; border-radius: 8px; border: 1px solid #dcdcdc; margin-top: 15px;">
                <p style="margin: 0; font-family: sans-serif; font-size: 14px; color: #31333F; font-weight: bold;">Developer: Chen Liu</p>
                <p style="margin: 0; font-family: sans-serif; font-size: 12px; color: #555;">Data Science & AI Leadership</p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("---")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Providers", f"{len(df):,}")
    k2.metric("High Propensity", f"{len(df[df['pred_class'] == 1]):,}")
    k3.metric("Unique Zips", f"{df['Rndrng_Prvdr_Zip5'].nunique():,}")
    k4.metric("Specialties", f"{df['Cleaned_Prvdr_Type'].nunique():,}")

    st.plotly_chart(plot_executive_map(df), width='stretch')

with chat_col:
    st.markdown("### AI Provider Targeting & Market Strategy")
    groq_api_key = st.secrets.get("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Please add your GROQ_API_KEY to Streamlit Secrets.")
        st.stop()

    # Feature flags: routing paths, in priority order.
    #   USE_LANGGRAPH -> LangGraph agent (src/agent_graph.py)
    #                    [plan -> execute -> grounding_check -> reflect -> finalize]
    #   USE_PLANNER   -> flat Groq tool-calling planner (src/planner.py)
    #   (always on)   -> legacy intent classifier + state-machine queue (src/router.py)
    # Each layer auto-falls-back to the next on failure, so the app always
    # has a working path and we can roll back without a deploy.
    USE_LANGGRAPH = bool(st.secrets.get("USE_LANGGRAPH", False))
    USE_PLANNER = bool(st.secrets.get("USE_PLANNER", True))
    # Debug instrumentation. When true, any silent fallback from the
    # LangGraph path surfaces the full traceback as an st.error(...) banner
    # in the chat column, instead of only being printed to Cloud logs.
    # Safe to leave off in prod; turn on only while diagnosing routing.
    DEBUG_ROUTING = bool(st.secrets.get("DEBUG_ROUTING", False))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_box = st.container(height=560, border=True)

    with chat_box:
        for message in st.session_state.messages:
            avatar = "🧬" if message["role"] == "assistant" else "👤"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

    # ----------------------------------------------------------------------
    # Per-intent answer builders (each returns a single markdown string)
    # ----------------------------------------------------------------------
    INTENT_LABEL = {
        "GENERAL": "general brand context",
        "OPPORTUNITY": "HCP opportunity targeting",
        "MARKETING": "marketing strategy",
        "NEWS": "competitor news and impact",
    }

    def _answer_opportunity(original_prompt: str, client: Groq) -> str:
        scorecard = get_hcp_scorecard(original_prompt, df)
        requested_npi = extract_npi(original_prompt)
        if not scorecard:
            if requested_npi:
                return f"NPI `{requested_npi}` was not found in the dataset."
            return "No high-propensity matches found for that filter."
        try:
            persona = build_system_prompt("data_analyst")
            analysis_prompt = f"""
                HCP PROFILE DATA:
                - NPI: {scorecard['npi']}
                - Location: {scorecard.get('city', 'N/A')}, {scorecard['state']}
                - Key Model Drivers: {scorecard['drivers']}

                INSTRUCTION:
                Briefly explain why these drivers make this HCP a high-priority
                target for Keytruda. Translate technical variables into strategic
                business terms. Limit to 2-3 sentences.
            """
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "system", "content": persona},
                          {"role": "user", "content": analysis_prompt}],
                temperature=0.3,
            )
            return _format_opportunity_markdown(scorecard, res.choices[0].message.content)
        except Exception as e:
            return _format_opportunity_markdown(scorecard, f"_(Insight unavailable: {str(e)[:80]})_")

    def _answer_marketing(original_prompt: str, client: Groq) -> str:
        scorecard = get_hcp_scorecard(original_prompt, df)
        requested_npi = extract_npi(original_prompt)
        if not scorecard:
            if requested_npi:
                return f"NPI `{requested_npi}` was not found in the dataset."
            return "Could not find a high-propensity target for strategy analysis."
        try:
            persona = build_system_prompt("marketing_specialist")
            mkt_prompt = f"""
                HCP CONTEXT:
                - Digital Adoption Score: {scorecard.get('digital_score', 'N/A')}
                - Days Since Last Engagement: {scorecard.get('last_engagement', 'N/A')}
                - Historically Preferred Channel: {scorecard.get('channel', 'Did not engage recently')}

                TASK:
                Suggest a 'Next Best Action' (NBA) strategy to increase adoption.
                Should we use the preferred channel or attempt a re-engagement?
                Limit to 2-3 professional sentences.
            """
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "system", "content": persona},
                          {"role": "user", "content": mkt_prompt}],
                temperature=0.4,
            )
            return _format_marketing_markdown(scorecard, res.choices[0].message.content)
        except Exception as e:
            return _format_marketing_markdown(scorecard, f"_(Strategy unavailable: {str(e)[:80]})_")

    def _answer_news(original_prompt: str, client: Groq) -> str:
        with st.spinner("Scanning competitor news and internal billing mix..."):
            brief = _cached_competitor_brief(original_prompt)

        news = brief.get("news") or []
        # Cache the just-displayed headlines so the user can paste one back
        # and get a quick summary on the next turn.
        st.session_state.last_news_items = news
        if news:
            news_ctx = "\n".join([
                f"- ({_short_date(n.get('date') or '')} "
                f"| {n.get('source') or '?'}) "
                f"{n.get('title') or ''} :: {(n.get('body') or '')[:240]}"
                for n in news[:8]
            ])
        else:
            news_ctx = "- (no live news available)"

        scope_label = brief.get("state") or "US (national)"
        try:
            persona = build_system_prompt("market_strategist")
            news_prompt = f"""
                USER QUESTION:
                {original_prompt}

                SCOPE: {scope_label}

                RECENT COMPETITOR HEADLINES (last ~month):
                {news_ctx}

                TASK:
                Respond with EXACTLY two short sections, each 2-3 sentences,
                and NOTHING ELSE. Do NOT add marketing-action bullets, do
                NOT reference any specific NPI, provider, city, preferred
                channel, digital adoption score, or engagement recency.
                Use this exact format:

                **📰 What happened recently:** <summary grounded in the
                headlines above; name the competitor(s); if news is sparse
                for this scope, say so plainly>

                **🔍 Competitor impact on Keytruda:** <concise read on
                potential impact to Keytruda's position / market share at
                this scope; stay at the brand-strategy level>
            """
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "system", "content": persona},
                          {"role": "user", "content": news_prompt}],
                temperature=0.3,
            )
            recommendation = res.choices[0].message.content
        except Exception as e:
            recommendation = f"_(Recommendation unavailable: {str(e)[:80]})_"

        return _format_news_markdown(brief, recommendation)

    def _answer_general(original_prompt: str, client: Groq) -> str:
        dataset_snapshot = (
            f"- Total providers in our targeting table: {len(df):,}\n"
            f"- High-propensity providers (pred_class == 1): {int((df['pred_class'] == 1).sum()):,}\n"
            f"- States covered: {df['Rndrng_Prvdr_State_Abrvtn'].nunique()}\n"
            f"- Provider specialties tracked: {df['Cleaned_Prvdr_Type'].nunique()}"
        )
        try:
            persona = build_system_prompt("brand_generalist")
            general_prompt = f"""
                USER QUESTION:
                {original_prompt}

                APPLICATION DATA SNAPSHOT (MerckAI_table.csv):
                {dataset_snapshot}

                INSTRUCTION:
                Answer the user's question as a Merck Keytruda brand advisor.
                If the question is about oncology science, IO mechanism of action,
                market dynamics, or general strategy, draw on your broader knowledge.
                If the question could be answered with the data snapshot above,
                use those numbers directly. Keep the response concise (under ~8
                bullets or a short paragraph).
            """
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "system", "content": persona},
                          {"role": "user", "content": general_prompt}],
                temperature=0.4,
            )
            return "### General Advisor\n\n" + res.choices[0].message.content
        except Exception as e:
            return f"_(General Advisor unavailable: {str(e)[:80]})_"

    INTENT_HANDLERS = {
        "GENERAL": _answer_general,
        "OPPORTUNITY": _answer_opportunity,
        "MARKETING": _answer_marketing,
        "NEWS": _answer_news,
    }

    def _answer_news_summary(item, client: Groq) -> str:
        """Pull the article body for a cached headline and ask the LLM for a
        short Keytruda-relevant summary. Falls back to the DuckDuckGo snippet
        if the page can't be fetched."""
        with st.spinner("Reading the article..."):
            article_text = _fetch_article_text(item.get("url") or "")
        # Fallback to the snippet DDG already gave us if the fetch fails.
        if not article_text:
            article_text = item.get("body") or ""

        if not article_text:
            return _format_news_summary_markdown(
                item,
                "_(Article content is unavailable — the publisher likely blocks "
                "automated access. Try opening the link directly.)_",
            )

        try:
            persona = build_system_prompt("market_strategist")
            summary_prompt = f"""
                ARTICLE TITLE: {item.get('title') or '(untitled)'}
                ARTICLE SOURCE: {item.get('source') or 'unknown'}
                ARTICLE DATE: {_short_date(item.get('date') or '')}

                ARTICLE CONTENT (may be truncated):
                {article_text}

                TASK:
                Summarize the article for a Merck Keytruda brand strategist.
                Output EXACTLY these three short sections, each 1-2 sentences,
                and NOTHING ELSE. Do not reference any NPI, provider, city,
                preferred channel, or engagement recency.

                **📝 What the article says:** <core facts from the content above>
                **🔍 Why it matters for Keytruda:** <potential impact on our
                position, pipeline, or competitive landscape>
                **🧭 What to watch next:** <the one thing a brand team should
                track as a follow-up>
            """
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "system", "content": persona},
                          {"role": "user", "content": summary_prompt}],
                temperature=0.2,
            )
            summary_text = res.choices[0].message.content
        except Exception as e:
            summary_text = f"_(Summary unavailable: {str(e)[:80]})_"

        return _format_news_summary_markdown(item, summary_text)

    def _answer_for_intent(intent: str, original_prompt: str, client: Groq) -> str:
        handler = INTENT_HANDLERS.get(intent)
        if handler is None:
            return f"_(Unhandled intent: {intent})_"
        return handler(original_prompt, client)

    # ----------------------------------------------------------------------
    # Planner-mode (Option 2) dispatch
    # ----------------------------------------------------------------------
    # The planner in `src/planner.py` emits an ordered list of tool calls
    # with already-extracted arguments. We dispatch each call to the
    # existing `_answer_*` handler, augmenting the original prompt with any
    # structured hints (state, NPI, competitor keys) so the downstream
    # prompt-based inference in `rag_engine` / `news_engine` still works
    # without any changes to those modules.
    def _augment_prompt_with_args(original_prompt: str, args: dict) -> str:
        extras: list[str] = []
        npi = args.get("npi")
        if isinstance(npi, (str, int)) and str(npi).strip():
            extras.append(f"NPI {str(npi).strip()}")
        state = args.get("state")
        if isinstance(state, str) and state.strip():
            extras.append(state.strip().upper())
        comps = args.get("competitors") or []
        if isinstance(comps, list) and comps:
            extras.append(" ".join(str(c) for c in comps))
        if not extras:
            return original_prompt
        return f"{original_prompt}  [planner hints: {' | '.join(extras)}]"

    def _dispatch_planner_call(
        name: str,
        args: dict,
        original_prompt: str,
        client: Groq,
    ) -> str | None:
        """Run one planner-emitted tool call and return its markdown string,
        or None if the tool name is unrecognized."""
        args = args or {}
        if name == "general_advisor":
            # If the planner answered in prose (no real tool_calls path in
            # `src/planner.py`), surface that text directly instead of
            # re-prompting the general advisor.
            prose = args.get("_prose")
            if isinstance(prose, str) and prose.strip():
                return "### General Advisor\n\n" + prose.strip()
            question = args.get("question") or original_prompt
            return _answer_general(question, client)
        if name == "get_hcp_opportunity":
            return _answer_opportunity(_augment_prompt_with_args(original_prompt, args), client)
        if name == "get_marketing_strategy":
            return _answer_marketing(_augment_prompt_with_args(original_prompt, args), client)
        if name == "get_competitor_news":
            return _answer_news(_augment_prompt_with_args(original_prompt, args), client)
        if name == "summarize_article":
            items = st.session_state.get("last_news_items") or []
            raw_idx = args.get("headline_index")
            try:
                idx = int(raw_idx)
            except (TypeError, ValueError):
                idx = -1
            if not items or not (0 <= idx < len(items)):
                return (
                    "_(I don't have a cached headline that matches. "
                    "Ask for competitor news first, then request a summary.)_"
                )
            return _answer_news_summary(items[idx], client)
        return None

    def _run_with_planner(user_prompt: str, client: Groq) -> str | None:
        """Plan + execute in planner mode. Returns assistant markdown, or
        None if the planner failed (caller should use the legacy path)."""
        cached = st.session_state.get("last_news_items") or []
        history = st.session_state.get("messages") or []
        with st.spinner("Planning..."):
            plan = plan_actions(
                user_prompt,
                groq_api_key,
                cached_headlines=cached,
                history=history,
            )
        if plan is None:
            # Hard failure inside the planner -> signal fallback.
            return None
        if not plan:
            plan = [{"name": "general_advisor", "args": {"question": user_prompt}, "id": ""}]

        sections: list[str] = []
        for call in plan:
            try:
                out = _dispatch_planner_call(
                    call.get("name") or "",
                    call.get("args") or {},
                    user_prompt,
                    client,
                )
            except Exception as e:  # noqa: BLE001
                out = f"_(Tool `{call.get('name')}` failed: {str(e)[:80]})_"
            if out:
                sections.append(out)

        if not sections:
            return None

        # Planner mode doesn't use the legacy confirmation queue; keep any
        # leftover state from legacy turns cleared so we don't re-ask.
        st.session_state.intent_queue = []
        st.session_state.original_prompt = ""
        st.session_state.pending_followup = False
        return "\n\n---\n\n".join(sections)

    # ----------------------------------------------------------------------
    # LangGraph-mode (Option 3) runner
    # ----------------------------------------------------------------------
    # The LangGraph agent in `src/agent_graph.py` wraps the planner above
    # in a 6-node state graph with two responsible-AI guardrails
    # (grounding_check and competitor_claim_check) and a reflection loop.
    # We use `graph.stream(..., stream_mode="updates")` and surface each
    # node's completion live in an `st.status` panel so the demo shows
    # the agent "thinking" step by step. On any hard failure the function
    # returns None, and the caller falls back to the native planner path.
    def _verify_npi_in_df(npi: str) -> bool:
        """Return True iff the NPI exists in the targeting table `df`."""
        try:
            s = df["Rndrng_NPI"].astype(str).str.strip()
            if (s == str(npi).strip()).any():
                return True
            return bool(
                (pd.to_numeric(df["Rndrng_NPI"], errors="coerce") == int(npi)).any()
            )
        except Exception:  # noqa: BLE001
            return True  # fail-open: don't block on lookup errors

    def _news_snapshot() -> list[dict]:
        """Ground truth for competitor_claim_check: whatever news items
        the most recent get_competitor_news dispatch cached."""
        return list(st.session_state.get("last_news_items") or [])

    def _format_graph_event(node: str, out: dict) -> str:
        """Render a single `graph.stream` update as one Markdown status line.
        Each retry re-emits the same nodes so the loop is visually obvious."""
        out = out or {}
        if node == "plan":
            if out.get("planner_failed"):
                err = (out.get("planner_error") or "").strip()
                if err:
                    return f"⚠️ **plan** — planner unavailable (`{err[:200]}`)"
                return "⚠️ **plan** — planner unavailable"
            plan = out.get("plan") or []
            if not plan:
                return "✓ **plan** — no tool calls"
            tool_names = ", ".join(
                str(c.get("name") or "?") for c in plan
            )
            return f"✓ **plan** — {len(plan)} tool call(s): `{tool_names}`"
        if node == "execute":
            n = len(out.get("outputs") or [])
            return f"✓ **execute** — {n} section(s) drafted"
        if node == "grounding_check":
            if out.get("grounding") == "HALLUCINATION":
                flagged = ", ".join(out.get("flagged_npis") or [])
                return f"⚠️ **grounding_check** — flagged NPI(s): {flagged}"
            return "✓ **grounding_check** — no hallucinated NPIs"
        if node == "competitor_claim_check":
            if out.get("competitor_claim") == "UNSUPPORTED":
                note = (out.get("competitor_claim_note") or "").strip()
                return (
                    f"⚠️ **competitor_claim_check** — unsupported claim"
                    + (f": {note[:140]}" if note else "")
                )
            return "✓ **competitor_claim_check** — claims OK"
        if node == "reflect":
            if out.get("reflection") == "RETRY":
                note = (out.get("reflection_note") or "").strip()
                return f"🔁 **reflect** — RETRY ({note[:120]})"
            return "✓ **reflect** — answer approved"
        if node == "finalize":
            return "✓ **finalize** — answer ready"
        return f"• **{node}**"

    def _report_routing_error(stage: str, exc: BaseException) -> None:
        """Log a LangGraph-path failure. Always prints to stdout (Cloud logs);
        when DEBUG_ROUTING is on, also surfaces the traceback in the UI so
        we can see *why* the graph silently fell back to the planner."""
        import traceback as _tb
        tb_text = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))
        print(f"[LangGraph:{stage}] {type(exc).__name__}: {exc}")
        print(tb_text)
        if DEBUG_ROUTING:
            st.error(
                f"**LangGraph `{stage}` failed** — falling back to planner.\n\n"
                f"`{type(exc).__name__}: {exc}`"
            )
            with st.expander("Traceback", expanded=False):
                st.code(tb_text, language="text")

    def _run_with_langgraph(user_prompt: str, client: Groq) -> str | None:
        """Build and stream the LangGraph agent for one user turn.
        Returns assistant markdown, or None on hard failure so the caller
        can fall back to `_run_with_planner`."""
        if DEBUG_ROUTING:
            st.info("DEBUG_ROUTING is ON — entering `_run_with_langgraph`.")
        try:
            from src.agent_graph import build_agent_graph
        except ImportError as e:
            _report_routing_error("import", e)
            return None

        def _planner_fn(u, h, hist):
            return plan_actions(
                u,
                groq_api_key,
                cached_headlines=h,
                history=hist,
            )

        def _dispatch_fn(name, args, original_prompt):
            return _dispatch_planner_call(name, args, original_prompt, client)

        try:
            graph = build_agent_graph(
                _planner_fn,
                _dispatch_fn,
                _verify_npi_in_df,
                groq_api_key,
                news_snapshot_fn=_news_snapshot,
            )
        except Exception as e:  # noqa: BLE001
            _report_routing_error("build", e)
            return None

        cached = st.session_state.get("last_news_items") or []
        history = st.session_state.get("messages") or []

        # `stream_mode="updates"` yields `{node_name: partial_state}` after
        # each node finishes. We merge them into `final_state` to recover
        # the complete state at the end (there's no single full-state
        # snapshot in "updates" mode).
        final_state: dict = {}
        try:
            with st.status("Agent graph: running…", expanded=True) as status:
                status.write("▶ **start** — routing user query through the graph")
                for event in graph.stream(
                    {
                        "user_prompt": user_prompt,
                        "cached_headlines": cached,
                        "history": history,
                        "retry_count": 0,
                    },
                    stream_mode="updates",
                ):
                    if not isinstance(event, dict):
                        continue
                    for node_name, node_output in event.items():
                        if node_name in ("__start__", "__end__"):
                            continue
                        status.write(_format_graph_event(node_name, node_output or {}))
                        if isinstance(node_output, dict):
                            final_state.update(node_output)
                status.update(
                    label="Agent graph: complete",
                    state="complete",
                    expanded=False,
                )
        except Exception as e:  # noqa: BLE001
            _report_routing_error("stream", e)
            return None

        final = final_state.get("final_answer")
        if not final:
            if DEBUG_ROUTING:
                st.warning(
                    "LangGraph stream finished but `final_answer` was empty. "
                    f"Keys in final_state: {list(final_state.keys())}"
                )
            return None

        # LangGraph mode, like planner mode, doesn't use the legacy queue.
        st.session_state.intent_queue = []
        st.session_state.original_prompt = ""
        st.session_state.pending_followup = False
        return final

    def _followup_question_md(just_answered: str, up_next: str) -> str:
        label_done = INTENT_LABEL.get(just_answered, just_answered.lower())
        label_next = INTENT_LABEL.get(up_next, up_next.lower())
        return (
            "\n\n---\n\n"
            f"_Hope the **{label_done}** above is helpful._ "
            f"I also noticed you asked about **{label_next}**. "
            "Would you like me to continue with that? "
            "Reply **yes** to continue, **no** to skip, "
            "or just ask a new question."
        )

    _YES_WORDS = {
        "y", "ye", "yes", "yea", "yeah", "yep", "yup", "ya", "yah", "ys",
        "sure", "ok", "okay", "k", "kk",
        "please", "pls", "plz", "yes please", "please do",
        "go", "go ahead", "continue", "next", "proceed",
        "do it", "sounds good", "affirmative", "affirm",
    }
    _NO_WORDS = {
        "n", "no", "nope", "nah", "naw", "skip", "stop",
        "not now", "later", "no thanks", "no thank you",
        "negative", "cancel",
    }
    _YES_PREFIX_CANDIDATES = [w for w in _YES_WORDS if len(w) >= 2 and " " not in w]
    _NO_PREFIX_CANDIDATES = [w for w in _NO_WORDS if len(w) >= 2 and " " not in w]

    def _classify_followup_reply(text: str) -> str:
        """Return YES / NO / NEW / UNCLEAR.

        UNCLEAR means the reply was too short/ambiguous to tell — the caller
        should NOT drop the pending queue; instead re-ask the user.
        """
        raw = (text or "").strip()
        t = raw.lower().rstrip(" .!?,;:")
        if not t:
            return "UNCLEAR"

        if t in _YES_WORDS:
            return "YES"
        if t in _NO_WORDS:
            return "NO"

        # Short-prefix tolerance so "ye" / "ok" / "nop" / "sur" still work.
        if len(t) <= 4:
            if any(w.startswith(t) for w in _YES_PREFIX_CANDIDATES):
                return "YES"
            if any(w.startswith(t) for w in _NO_PREFIX_CANDIDATES):
                return "NO"
            # Very short and not a recognizable affirmation → ambiguous,
            # don't treat as a fresh query (avoids wiping the queue on typos).
            return "UNCLEAR"

        # Longer text: accept "yes, please continue" or "no, ask about TX".
        first_token = t.split()[0].strip(",.!?:;") if t else ""
        if first_token in _YES_WORDS and len(t) <= 60:
            return "YES"
        if first_token in _NO_WORDS and len(t) <= 60:
            return "NO"
        return "NEW"

    # ----------------------------------------------------------------------
    # State-machine chat handler
    # ----------------------------------------------------------------------
    for key, default in (
        ("intent_queue", []),
        ("original_prompt", ""),
        ("pending_followup", False),
        ("last_news_items", []),
    ):
        if key not in st.session_state:
            st.session_state[key] = default

    def _maybe_answer_news_summary(user_prompt: str, client: Groq):
        """Return an assistant markdown string if the user is asking to
        summarize one of the recently-shown headlines, else None.

        Triggers in two ways:
          1. User message clearly matches a cached headline (paste/partial).
          2. User message contains an explicit 'summarize/tell me more' cue
             AND cached headlines exist; in that case we match either the
             explicit reference or fall back to the first cached headline.
        """
        items = st.session_state.get("last_news_items") or []
        if not items:
            return None

        matched = _find_news_item_for_query(user_prompt, items)
        if matched is None and _looks_like_summary_request(user_prompt):
            # Explicit cue, no obvious title match → use the first cached headline.
            matched = items[0]
        if matched is None:
            return None
        return _answer_news_summary(matched, client)

    def _start_fresh(user_prompt: str, client: Groq) -> str:
        """Detect intents, answer the first, queue the rest, return assistant markdown."""
        # Short-circuit: if the user is clearly asking us to summarize a
        # previously-shown headline, do that instead of re-running routing.
        summary_answer = _maybe_answer_news_summary(user_prompt, client)
        if summary_answer is not None:
            # Clear any pending queue — they've pivoted to article summarization.
            st.session_state.intent_queue = []
            st.session_state.original_prompt = ""
            st.session_state.pending_followup = False
            return summary_answer

        with st.spinner("Routing..."):
            intents = get_intents(user_prompt, groq_api_key)
        first, remaining = intents[0], intents[1:]
        content = _answer_for_intent(first, user_prompt, client)
        if remaining:
            content += _followup_question_md(first, remaining[0])
            st.session_state.intent_queue = remaining
            st.session_state.original_prompt = user_prompt
            st.session_state.pending_followup = True
        else:
            st.session_state.intent_queue = []
            st.session_state.original_prompt = ""
            st.session_state.pending_followup = False
        return content

    def _continue_queue(client: Groq) -> str:
        """Answer the next queued intent against the stored original prompt."""
        next_intent = st.session_state.intent_queue.pop(0)
        content = _answer_for_intent(
            next_intent,
            st.session_state.original_prompt,
            client,
        )
        if st.session_state.intent_queue:
            content += _followup_question_md(next_intent, st.session_state.intent_queue[0])
            st.session_state.pending_followup = True
        else:
            st.session_state.original_prompt = ""
            st.session_state.pending_followup = False
        return content

    if prompt := st.chat_input("Ask about opportunities, marketing, competitor news, or anything Keytruda..."):
        # Keep every per-turn UI element (user bubble, st.status progress
        # panel, spinners, assistant bubble) rendered *inside* chat_box
        # so the whole chat column stays within a fixed height budget
        # and visually lines up with the left BI column.
        with chat_box:
            st.chat_message("user", avatar="👤").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            client = Groq(api_key=groq_api_key)

            # Primary paths, in priority order. Each returns None on hard
            # failure so we fall through to the next layer. We skip both
            # modern paths if the user is mid-followup in a legacy queue so
            # the yes/no UX stays consistent.
            assistant_content = None
            if not st.session_state.pending_followup:
                if USE_LANGGRAPH:
                    assistant_content = _run_with_langgraph(prompt, client)
                if assistant_content is None and USE_PLANNER:
                    assistant_content = _run_with_planner(prompt, client)

            if assistant_content is not None:
                # LangGraph or planner handled this turn.
                st.session_state.messages.append({"role": "assistant", "content": assistant_content})
                with st.chat_message("assistant", avatar="🧬"):
                    st.markdown(assistant_content)
                st.stop()

            # ---- Legacy path (automatic fallback when all modern paths
            # returned None, or when both feature flags are off, or when
            # we're mid-followup). ----
            if st.session_state.pending_followup:
                reply_kind = _classify_followup_reply(prompt)
                if reply_kind == "YES":
                    st.session_state.pending_followup = False
                    assistant_content = _continue_queue(client)
                elif reply_kind == "NO":
                    st.session_state.intent_queue = []
                    st.session_state.original_prompt = ""
                    st.session_state.pending_followup = False
                    assistant_content = (
                        "Understood — I'll drop that follow-up. "
                        "What would you like to look at next?"
                    )
                elif reply_kind == "UNCLEAR":
                    # Keep the queue + original_prompt intact so a typo like "Ye"
                    # doesn't wipe context. Re-prompt the user for a clear answer.
                    next_intent = (
                        st.session_state.intent_queue[0]
                        if st.session_state.intent_queue
                        else None
                    )
                    if next_intent:
                        label_next = INTENT_LABEL.get(next_intent, next_intent.lower())
                        assistant_content = (
                            f"I didn't quite catch that. Did you mean **yes** to continue "
                            f"with **{label_next}** for your original question"
                            + (
                                f" _(\"{st.session_state.original_prompt}\")_"
                                if st.session_state.original_prompt
                                else ""
                            )
                            + ", or **no** to skip? You can also just type a new question."
                        )
                    else:
                        st.session_state.pending_followup = False
                        assistant_content = (
                            "I didn't quite catch that — could you rephrase what you'd "
                            "like me to look into?"
                        )
                else:
                    # A new question -- drop any pending queue and start fresh.
                    st.session_state.intent_queue = []
                    st.session_state.original_prompt = ""
                    st.session_state.pending_followup = False
                    assistant_content = _start_fresh(prompt, client)
            else:
                assistant_content = _start_fresh(prompt, client)

            st.session_state.messages.append({"role": "assistant", "content": assistant_content})
            with st.chat_message("assistant", avatar="🧬"):
                st.markdown(assistant_content)