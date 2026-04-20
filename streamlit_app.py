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

def _format_news_markdown(brief, recommendation_text=None):
    """Render the competitor-news view: inferred scope, internal billing mix,
    live headlines, and an AI-synthesized marketing recommendation."""
    state = brief.get("state")
    scope_label = f"State: {state}" if state else "Scope: National"
    lines = [
        f"### 📰 Competitor Intelligence ({scope_label})",
    ]

    share = brief.get("share") or {}
    if share:
        lines.append("**Internal billing-share mix (avg across providers):**")
        sorted_share = sorted(share.items(), key=lambda kv: (kv[1] if kv[1] == kv[1] else -1), reverse=True)
        for label, val in sorted_share:
            try:
                val_fmt = f"{float(val):.2f}"
            except (TypeError, ValueError):
                val_fmt = "n/a"
            lines.append(f"- **{label}:** {val_fmt}")

    news = brief.get("news")
    if news is None:
        lines.append("_Live news search unavailable (package missing or network blocked)._")
    elif not news:
        lines.append("_No recent articles returned for the inferred competitors._")
    else:
        lines.append("**Recent headlines:**")
        for item in news[:8]:
            title = item.get("title") or "(untitled)"
            source = item.get("source") or ""
            date = item.get("date") or ""
            url = item.get("url") or ""
            meta_bits = " · ".join([b for b in [source, date] if b])
            if url:
                lines.append(f"- [{title}]({url}) — {meta_bits}")
            else:
                lines.append(f"- {title} — {meta_bits}")

    top_hcp = brief.get("top_hcp")
    if top_hcp:
        lines.append(
            f"**Anchor HCP for marketing ({scope_label}):** "
            f"NPI {top_hcp['npi']} — {top_hcp.get('city', 'N/A')}, {top_hcp['state']} · "
            f"Propensity {top_hcp['score']:.1%} · Preferred Channel: {top_hcp.get('channel', 'Unknown')}"
        )

    md = "\n\n".join(lines)
    if recommendation_text:
        md += f"\n\n🎯 **Strategic Recommendation:** {recommendation_text}"
    return md


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
        f"### 📈 Marketing Strategy: NPI {scorecard['npi']}",
        _lookup_label(scorecard),
        f"**Location:** {scorecard.get('city', 'N/A')}, {scorecard['state']} | **Propensity:** {scorecard['score']:.1%}",
        "**Engagement Metrics:**",
        f"- **Digital Adoption Score:** {digital_score:.2f}",
        f"- **Last Engagement:** {last_engagement} days ago",
        f"- **Preferred Channel:** {scorecard.get('channel', 'Unknown')}",
    ]
    md = "\n\n".join(lines)
    if strategy_text:
        md += f"\n\n🎯 **Next Best Action:** {strategy_text}"
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
main_col, chat_col = st.columns([2.2, 1], gap="medium")

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
    st.markdown("### AI Brand Strategy Assistant")
    groq_api_key = st.secrets.get("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Please add your GROQ_API_KEY to Streamlit Secrets.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_box = st.container(height=650, border=True)

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
                model="llama-3.3-70b-versatile",
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
                model="llama-3.3-70b-versatile",
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

        share = brief.get("share") or {}
        share_ctx = "\n".join([f"- {k}: {v}" for k, v in share.items()]) or "- (no CSV share data)"

        news = brief.get("news") or []
        if news:
            news_ctx = "\n".join([
                f"- ({n.get('date') or 'n/d'}) [{n.get('source') or '?'}] "
                f"{n.get('title') or ''} :: {(n.get('body') or '')[:240]}"
                for n in news[:8]
            ])
        else:
            news_ctx = "- (no live news available)"

        top_hcp = brief.get("top_hcp")
        if top_hcp:
            anchor_ctx = (
                f"NPI {top_hcp['npi']} in {top_hcp.get('city', 'N/A')}, {top_hcp['state']}; "
                f"propensity {float(top_hcp['score']):.3f}; "
                f"preferred channel {top_hcp.get('channel', 'Unknown')}; "
                f"digital adoption {top_hcp.get('digital_score', 0)}; "
                f"days since last engagement {top_hcp.get('last_engagement', 0)}."
            )
        else:
            anchor_ctx = "(no anchor HCP available for this scope)"

        scope_label = brief.get("state") or "US (national)"
        try:
            persona = build_system_prompt("market_strategist")
            news_prompt = f"""
                USER QUESTION:
                {original_prompt}

                SCOPE: {scope_label}

                INTERNAL BILLING-SHARE MIX (average across providers in scope):
                {share_ctx}

                RECENT COMPETITOR HEADLINES (last ~month):
                {news_ctx}

                ANCHOR HCP FOR MARKETING ACTIONS:
                {anchor_ctx}

                TASK:
                In 4-6 short bullet points, summarize the most important
                competitor movement for this scope and recommend a concrete
                marketing response for Keytruda. When possible, reference
                the anchor HCP's preferred channel and engagement recency.
                Call out uncertainty if the live news is sparse.
            """
            res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
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
                model="llama-3.3-70b-versatile",
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

    def _answer_for_intent(intent: str, original_prompt: str, client: Groq) -> str:
        handler = INTENT_HANDLERS.get(intent)
        if handler is None:
            return f"_(Unhandled intent: {intent})_"
        return handler(original_prompt, client)

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
        "y", "yes", "yeah", "yep", "ya", "yup",
        "sure", "ok", "okay", "please", "yes please",
        "go ahead", "continue", "next", "do it", "go", "sounds good",
    }
    _NO_WORDS = {
        "n", "no", "nope", "nah", "skip", "stop", "not now", "later",
        "no thanks", "no thank you",
    }

    def _classify_followup_reply(text: str) -> str:
        """YES / NO / NEW. NEW means treat as a fresh query."""
        t = (text or "").strip().lower().rstrip(" .!?")
        if t in _YES_WORDS:
            return "YES"
        if t in _NO_WORDS:
            return "NO"
        # Allow short prefixes like "yes, please continue" or "sure, go on".
        # Strip punctuation from the first token so "sure," still matches.
        first_token = t.split()[0].strip(",.!?:;") if t else ""
        if first_token in _YES_WORDS and len(t) <= 40:
            return "YES"
        if first_token in _NO_WORDS and len(t) <= 40:
            return "NO"
        return "NEW"

    # ----------------------------------------------------------------------
    # State-machine chat handler
    # ----------------------------------------------------------------------
    for key, default in (
        ("intent_queue", []),
        ("original_prompt", ""),
        ("pending_followup", False),
    ):
        if key not in st.session_state:
            st.session_state[key] = default

    def _start_fresh(user_prompt: str, client: Groq) -> str:
        """Detect intents, answer the first, queue the rest, return assistant markdown."""
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
        with chat_box:
            st.chat_message("user", avatar="👤").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        client = Groq(api_key=groq_api_key)

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
            else:
                # A new question -- drop any pending queue and start fresh.
                st.session_state.intent_queue = []
                st.session_state.original_prompt = ""
                st.session_state.pending_followup = False
                assistant_content = _start_fresh(prompt, client)
        else:
            assistant_content = _start_fresh(prompt, client)

        st.session_state.messages.append({"role": "assistant", "content": assistant_content})
        with chat_box:
            with st.chat_message("assistant", avatar="🧬"):
                st.markdown(assistant_content)