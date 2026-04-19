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
from src.prompts import SYSTEM_PERSONAS

# --- 1. SETUP & DATA CLEANING ---
st.set_page_config(page_title="Merck Data Science Hub", layout="wide")

@st.cache_data
def load_data():
    data_path = os.path.join(current_dir, 'data', 'raw', 'MerckAI_table.csv')
    if not os.path.exists(data_path):
        st.error(f"Data file not found at: {data_path}")
        st.stop()
        
    df = pd.read_csv(data_path)
    
    # DATA HYGIENE: Force numeric conversion for SHAP columns 
    # (Fixes the hidden apostrophe prefix issue)
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

    # Geographic Heatmap
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

        # 2. Assistant Logic
        with chat_box:
            with st.chat_message("assistant", avatar="🧬"):
                if intent == "OPPORTUNITY":
                    scorecard = get_hcp_scorecard(prompt, df)
                    
                    if scorecard:
                        # UI: NPI Header
                        st.markdown(f"### 🎯 NPI: {scorecard['npi']}")
                        
                        # UPDATED: Location (City, State)
                        st.success(f"**Location:** {scorecard['city']}, {scorecard['state']} | **Type:** {scorecard['type']}")
                        
                        st.markdown(f"**Opportunity Scorecard:**")
                        st.write(f"- **Propensity Score:** {scorecard['score']:.1%}")
                        
                        # UPDATED: Metric Label
                        st.write(f"- **Avg Medicare Payment per Person:** ${scorecard['payment']:,.2f}")
                        
                        st.write(f"- **Top Model Drivers:** {scorecard['drivers']}")
                        
                        # --- AI INSIGHT (Llama 3.3 Versatile) ---
                        try:
                            client = Groq(api_key=groq_api_key)
                            persona = SYSTEM_PERSONAS.get("data_analyst", "You are a Merck Lead Data Scientist.")
                            
                            analysis_prompt = f"""
                            HCP PROFILE DATA:
                            - NPI: {scorecard['npi']}
                            - Location: {scorecard['city']}, {scorecard['state']}
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
                                    {"role": "user", "content": analysis_prompt}
                                ],
                                temperature=0.3
                            )
                            st.info(f"💡 **AI Insight:** {res.choices[0].message.content}")
                        except Exception as e:
                            st.error(f"Insight Error: {str(e)[:50]}")
                            st.warning("⚠️ High-level drivers identified, but AI summary was unavailable.")
                    else:
                        st.error("No high-propensity matches found for that specific query.")

                elif intent == "MARKETING":
                    st.markdown("### 📈 Marketing Strategy")
                    st.warning("Analyzing engagement paths and omnichannel strategy...")

                elif intent == "NEWS":
                    st.info("📰 Scanning competitive landscape and clinical trial news...")

        st.session_state.messages.append({"role": "assistant", "content": f"Handled as: {intent}"})