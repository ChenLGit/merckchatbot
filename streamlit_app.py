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

from src.router import get_intent
from src.visualizations import plot_executive_map
from src.rag_engine import get_hcp_scorecard

try:
    from src.prompts import SYSTEM_PERSONAS
except ImportError:
    SYSTEM_PERSONAS = {}

# ---------------------------------
# Helpers: Format Outputs
# ---------------------------------
def _format_opportunity_markdown(scorecard, insight_text=None):
    lines = [
        f"### 🎯 NPI: {scorecard['npi']}",
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

def _format_marketing_markdown(scorecard, strategy_text=None):
    """Renders the marketing strategy view using synthetic engagement data."""
    lines = [
        f"### 📈 Marketing Strategy: NPI {scorecard['npi']}",
        f"**Location:** {scorecard.get('city', 'N/A')}, {scorecard['state']}",
        "**Engagement Metrics:**",
        f"- **Digital Adoption Score:** {scorecard.get('digital_score', 0):.2f}",
        f"- **Last Engagement:** {scorecard.get('last_engagement', 0)} days ago",
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
                <p style="margin: 0; font-family: sans-serif; font-size: 14px; color: #31333F; font-weight: bold;">Lead: Chen Liu</p>
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

        with st.spinner("Routing..."):
            intent = get_intent(prompt, groq_api_key)

        assistant_content = ""
        client = Groq(api_key=groq_api_key)

        if intent == "OPPORTUNITY":
            scorecard = get_hcp_scorecard(prompt, df)
            if scorecard:
                try:
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
                        messages=[{"role": "system", "content": persona}, {"role": "user", "content": analysis_prompt}],
                        temperature=0.3,
                    )
                    assistant_content = _format_opportunity_markdown(scorecard, res.choices[0].message.content)
                except Exception as e:
                    assistant_content = _format_opportunity_markdown(scorecard, f"_(Insight unavailable: {str(e)[:80]})_")
            else:
                assistant_content = "No high-propensity matches found for that filter."

        elif intent == "MARKETING":
            # Re-use the RAG engine to find the same top target, then apply marketing logic
            scorecard = get_hcp_scorecard(prompt, df)
            if scorecard:
                try:
                    persona = SYSTEM_PERSONAS.get("marketing_specialist", "You are a Merck Marketing Lead.")
                    mkt_prompt = f"""
                        HCP CONTEXT:
                        - Digital Adoption Score: {scorecard.get('digital_score', 0)}
                        - Days Since Last Engagement: {scorecard.get('last_engagement', 0)}
                        - Historically Preferred Channel: {scorecard.get('channel', 'Unknown')}

                        TASK:
                        Suggest a 'Next Best Action' (NBA) strategy to increase adoption. 
                        Should we use the preferred channel or attempt a re-engagement?
                        Limit to 2-3 professional sentences.
                    """
                    res = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "system", "content": persona}, {"role": "user", "content": mkt_prompt}],
                        temperature=0.4,
                    )
                    assistant_content = _format_marketing_markdown(scorecard, res.choices[0].message.content)
                except Exception as e:
                    assistant_content = _format_marketing_markdown(scorecard, f"_(Strategy unavailable: {str(e)[:80]})_")
            else:
                assistant_content = "Could not find a high-propensity target for strategy analysis."

        elif intent == "NEWS":
            assistant_content = "### 📰 News scan\n\nScanning competitive landscape and clinical trial results..."
        
        else:
            assistant_content = f"_(Unhandled intent: {intent})_"

        st.session_state.messages.append({"role": "assistant", "content": assistant_content})
        with chat_box:
            with st.chat_message("assistant", avatar="🧬"):
                st.markdown(assistant_content)