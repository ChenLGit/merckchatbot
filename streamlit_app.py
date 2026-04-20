import sys
import os
import streamlit as st
import pandas as pd
from groq import Groq

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from src.router import get_intent
from src.visualizations import plot_executive_map
from src.rag_engine import get_hcp_scorecard

# SAFETY CHECK: Import personas with a fallback to prevent KeyError crashes during startup
try:
    from src.prompts import SYSTEM_PERSONAS
except ImportError:
    SYSTEM_PERSONAS = {}


def _format_opportunity_markdown(scorecard, insight_text=None):
    """Single markdown blob so chat history survives Streamlit reruns."""
    lines = [
        f"### 🎯 NPI: {scorecard['npi']}",
        (
            f"**Location:** {scorecard.get('city', 'N/A')}, {scorecard['state']} | "
            f"**Type:** {scorecard['type']}"
        ),
        "**Opportunity Scorecard:**",
        f"- **Propensity Score:** {scorecard['score']:.1%}",
        f"- **Avg Medicare Payment per Person:** ${scorecard['payment']:,.2f}",
        f"- **Top Model Drivers:** {scorecard['drivers']}",
    ]
    md = "\n\n".join(lines)
    if insight_text:
        md += f"\n\n💡 **AI Insight:** {insight_text}"
    return md


# --- 1. SETUP & DATA CLEANING ---
st.set_page_config(page_title="Merck Data Science Hub", layout="wide")

@st.cache_data
def load_data():
    # Points to the raw CSV data
    data_path = os.path.join(current_dir, 'data', 'raw', 'MerckAI_table.csv')
    if not os.path.exists(data_path):
        st.error(f"Data file not found at: {data_path}")
        st.stop()

    df = pd.read_csv(data_path)

    # DATA HYGIENE: Force numeric conversion for SHAP columns
    # (Fixes the hidden apostrophe prefix issue found in raw CSV)
    shap_cols = [c for c in df.columns if c.startswith('SHAP_')]
    for col in shap_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace("'", ""), errors='coerce').fillna(0)

    return df

df = load_data()

# --- 2. UI LAYOUT (BI on Left, AI on Right) ---
main_col, chat_col = st.columns([2.2, 1], gap="medium")

# --- LEFT COLUMN: BI & VISUALIZATIONS ---
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
                <p style="margin: 0; font-family: sans-serif; font-size: 14px; color: #31333F; font-weight: bold;">Lead: Chen Liu</p>
                <p style="margin: 0; font-family: sans-serif; font-size: 12px; color: #555;">Data Science & AI Leadership</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # BI KPI Layer
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Providers", f"{len(df):,}")
    k2.metric("High Propensity", f"{len(df[df['pred_class'] == 1]):,}")
    k3.metric("Unique Zips", f"{df['Rndrng_Prvdr_Zip5'].nunique():,}")
    k4.metric("Specialties", f"{df['Cleaned_Prvdr_Type'].nunique():,}")

    # Geographic Heatmap (2026 stretch standard)
    st.plotly_chart(plot_executive_map(df), width='stretch')

# --- RIGHT COLUMN: AI STRATEGY ASSISTANT ---
with chat_col:
    st.markdown("### 🤖 Strategy Assistant")

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

    if prompt := st.chat_input("Analyze HCP opportunity..."):
        with chat_box:
            st.chat_message("user", avatar="👤").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 1. Intent Routing
        with st.spinner("Routing..."):
            intent = get_intent(prompt, groq_api_key)

        # 2. Assistant logic — persist full reply text so reruns replay the whole thread.
        assistant_content = ""
        if intent == "OPPORTUNITY":
            scorecard = get_hcp_scorecard(prompt, df)
            if scorecard:
                insight_text = None
                try:
                    client = Groq(api_key=groq_api_key)
                    persona = SYSTEM_PERSONAS.get("data_analyst", "You are a Merck Lead Data Scientist.")
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
                        messages=[
                            {"role": "system", "content": persona},
                            {"role": "user", "content": analysis_prompt},
                        ],
                        temperature=0.3,
                    )
                    insight_text = res.choices[0].message.content
                except Exception as e:
                    insight_text = f"_(Insight unavailable: {str(e)[:80]})_"
                assistant_content = _format_opportunity_markdown(scorecard, insight_text)
            else:
                assistant_content = (
                    "**OPPORTUNITY**\n\n"
                    "No high-propensity matches found for that specific state or filter."
                )
        elif intent == "MARKETING":
            assistant_content = (
                "### 📈 Marketing Strategy\n\n"
                "Analyzing engagement paths and omnichannel messaging frequency… "
                "_(Placeholder — connect your marketing workflow here.)_"
            )
        elif intent == "NEWS":
            assistant_content = (
                "### 📰 News scan\n\n"
                "Scanning competitive landscape and clinical trial results… "
                "_(Placeholder — connect your news feed here.)_"
            )
        else:
            assistant_content = f"_(Unhandled intent: {intent})_"

        st.session_state.messages.append({"role": "assistant", "content": assistant_content})

        with chat_box:
            with st.chat_message("assistant", avatar="🧬"):
                st.markdown(assistant_content)
