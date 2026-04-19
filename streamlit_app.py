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
from src.rag_engine import get_hcp_scorecard        # Added RAG Engine
from src.prompts import SYSTEM_PERSONAS           # Added Personas

# --- 1. SETUP PAGE CONFIG ---
st.set_page_config(page_title="Merck Data Science Hub", layout="wide")

# --- 2. DATA LOADING ---
@st.cache_data
def load_data():
    data_path = os.path.join(current_dir, 'data', 'raw', 'MerckAI_table.csv')
    if not os.path.exists(data_path):
        st.error(f"Data file not found at: {data_path}")
        st.stop()
    return pd.read_csv(data_path)

df = load_data()

# --- 3. LAYOUT DEFINITION ---
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
                <p style="margin: 0; font-family: sans-serif; font-size: 14px; color: #31333F; font-weight: bold;">Developed by: Chen Liu</p>
                <p style="margin: 0; font-family: sans-serif; font-size: 12px; color: #555;">chen.liu1010@gmail.com</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # KPIs
    total_providers = len(df)
    predicted_yes = len(df[df['pred_class'] == 1])
    unique_zips = df['Rndrng_Prvdr_Zip5'].nunique()
    total_types = df['Cleaned_Prvdr_Type'].nunique()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Providers", f"{total_providers:,}")
    kpi2.metric("High Propensity", f"{predicted_yes:,}")
    kpi3.metric("Unique Zip5s", f"{unique_zips:,}")
    kpi4.metric("Specialties", f"{total_types:,}")

    st.plotly_chart(plot_executive_map(df), use_container_width=True)

# --- RIGHT COLUMN: AI CONVERSATIONAL INTERFACE ---
with chat_col:
    st.markdown("### 🤖 Strategy Assistant")
    
    groq_api_key = st.secrets.get("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Missing GROQ_API_KEY.")
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
        
        # 1. Get Intent
        with st.spinner("Routing..."):
            intent = get_intent(prompt, groq_api_key)

        # 2. Assistant Response Logic
        with chat_box:
            with st.chat_message("assistant", avatar="🧬"):
                if intent == "OPPORTUNITY":
                    # Call the RAG Engine to pull data + SHAP
                    scorecard = get_hcp_scorecard(prompt, df)
                    
                    if scorecard:
                        st.markdown("### 🎯 Target Opportunity Found")
                        st.success(f"**Target:** {scorecard['type']} | **State:** {scorecard['state']}")
                        
                        st.markdown(f"""
                        **Opportunity Scorecard:**
                        - **Propensity Score:** {scorecard['score']:.1%}
                        - **Medicare Payment Volume:** ${scorecard['payment']:,.0f}
                        - **Key Model Drivers:** {scorecard['drivers']}
                        """)
                        
                        # Generate AI Insight explaining the SHAP drivers
                        client = Groq(api_key=groq_api_key)
                        explanation = client.chat.completions.create(
                            model="llama3-8b-8192",
                            messages=[
                                {"role": "system", "content": SYSTEM_PERSONAS["data_analyst"]},
                                {"role": "user", "content": f"Explain why these SHAP drivers make this HCP a high-priority target: {scorecard['drivers']}"}
                            ]
                        )
                        st.info(f"💡 **AI Insight:** {explanation.choices[0].message.content}")
                    else:
                        st.error("No specific opportunity found for that query.")

                elif intent == "MARKETING":
                    st.markdown("### 📈 Marketing Strategy")
                    st.warning("Analyzing Omnichannel engagement paths...")
                    st.markdown("Optimizing touchpoint frequency based on physician response patterns.")

                elif intent == "NEWS":
                    st.markdown("### 📰 Market Intelligence")
                    st.info("Scanning competitive landscape (BMS, Opdivo, Roche)...")
                    st.markdown("Accessing real-time clinical trial updates and FDA filings.")

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": f"Handled as: {intent}"})