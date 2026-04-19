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

# Import custom modules
from src.router import get_intent
from src.visualizations import plot_executive_map
from src.rag_engine import get_hcp_scorecard
from src.prompts import SYSTEM_PERSONAS

# --- 1. SETUP PAGE CONFIG ---
st.set_page_config(page_title="Merck Data Science Hub", layout="wide")

# --- 2. DATA LOADING ---
@st.cache_data
def load_data():
    # Points to your specific data structure
    data_path = os.path.join(current_dir, 'data', 'raw', 'MerckAI_table.csv')
    if not os.path.exists(data_path):
        st.error(f"Data file not found at: {data_path}")
        st.stop()
    return pd.read_csv(data_path)

df = load_data()

# --- 3. LAYOUT DEFINITION (Fixed Side-by-Side) ---
main_col, chat_col = st.columns([2.2, 1], gap="medium")

# --- LEFT COLUMN: BI & MAP ---
with main_col:
    # Professional Header
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
                <p style="margin: 0; font-family: sans-serif; font-size: 14px; color: #31333F; font-weight: bold;">Developed by: Chen Liu</p>
                <p style="margin: 0; font-family: sans-serif; font-size: 12px; color: #555;">chen.liu1010@gmail.com</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # BI KPI Row
    total_providers = len(df)
    predicted_yes = len(df[df['pred_class'] == 1])
    unique_zips = df['Rndrng_Prvdr_Zip5'].nunique()
    total_types = df['Cleaned_Prvdr_Type'].nunique()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Providers", f"{total_providers:,}")
    k2.metric("High Propensity", f"{predicted_yes:,}")
    k3.metric("Unique Zip5s", f"{unique_zips:,}")
    k4.metric("Specialties", f"{total_types:,}")

    # Map Visualization
    st.plotly_chart(plot_executive_map(df), use_container_width=True)

# --- RIGHT COLUMN: AI CONVERSATIONAL INTERFACE ---
with chat_col:
    st.markdown("### 🤖 Strategy Assistant")
    
    groq_api_key = st.secrets.get("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Missing GROQ_API_KEY in secrets.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat history container with fixed height
    chat_box = st.container(height=650, border=True)

    with chat_box:
        for message in st.session_state.messages:
            avatar = "🧬" if message["role"] == "assistant" else "👤"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

    # Input Logic
    if prompt := st.chat_input("Analyze HCP opportunity..."):
        with chat_box:
            st.chat_message("user", avatar="👤").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Routing..."):
            try:
                intent = get_intent(prompt, groq_api_key)
            except:
                intent = "OPPORTUNITY"

        with chat_box:
            with st.chat_message("assistant", avatar="🧬"):
                if intent == "OPPORTUNITY":
                    scorecard = get_hcp_scorecard(prompt, df)
                    
                    if scorecard:
                        # NEW: NPI is now the main header
                        st.markdown(f"### 🎯 NPI: {scorecard['npi']}")
                        st.success(f"**State:** {scorecard['state']} | **Type:** {scorecard['type']}")
                        
                        st.markdown(f"""
                        **Opportunity Scorecard:**
                        - **Propensity Score:** {scorecard['score']:.1%}
                        - **Medicare Payment Volume:** ${scorecard['payment']:,.0f}
                        - **Key Model Drivers:** {scorecard['drivers']}
                        """)
                        
                        # (AI Insight section remains the same)
                        
                        # --- STABLE AI INSIGHT ---
                        try:
                            # Clean drivers string for the LLM
                            clean_drivers = str(scorecard['drivers']).replace("{", "").replace("}", "")
                            persona = SYSTEM_PERSONAS.get("data_analyst", "You are a Merck Lead Data Scientist.")
                            
                            client = Groq(api_key=groq_api_key)
                            explanation = client.chat.completions.create(
                                model="llama3-8b-8192",
                                messages=[
                                    {"role": "system", "content": persona},
                                    {"role": "user", "content": f"Briefly explain why these clinical drivers matter for Keytruda targeting: {clean_drivers}"}
                                ]
                            )
                            st.info(f"💡 **AI Insight:** {explanation.choices[0].message.content}")
                        except Exception as e:
                            st.warning("⚠️ High-level drivers identified, but AI summary was unavailable.")
                    else:
                        st.error("No specific opportunity found.")

                elif intent == "MARKETING":
                    st.markdown("### 📈 Marketing Strategy")
                    st.warning("Analyzing engagement tactics and omnichannel frequency...")

                elif intent == "NEWS":
                    st.markdown("### 📰 Market Intelligence")
                    st.info("Scanning competitive landscape and clinical trial news...")

        st.session_state.messages.append({"role": "assistant", "content": f"Routed as: {intent}"})