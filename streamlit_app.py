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

# --- 1. SETUP & DATA ---
st.set_page_config(page_title="Merck Data Science Hub", layout="wide")

@st.cache_data
def load_data():
    data_path = os.path.join(current_dir, 'data', 'raw', 'MerckAI_table.csv')
    return pd.read_csv(data_path)

df = load_data()

# --- 2. UI LAYOUT ---
main_col, chat_col = st.columns([2.2, 1], gap="medium")

with main_col:
    # Professional Header
    st.markdown("""<div style="font-family: sans-serif; font-size: 3.8rem; font-weight: bold; color: #00857c;">Merck Keytruda</div>""", unsafe_allow_html=True)
    st.markdown("---")
    
    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Providers", f"{len(df):,}")
    k2.metric("High Propensity", f"{len(df[df['pred_class']==1]):,}")
    k3.metric("Unique Zips", f"{df['Rndrng_Prvdr_Zip5'].nunique():,}")
    k4.metric("Specialties", f"{df['Cleaned_Prvdr_Type'].nunique():,}")

    st.plotly_chart(plot_executive_map(df), use_container_width=True)

# --- 3. CHAT PANEL (RIGHT SIDE) ---
with chat_col:
    st.markdown("### 🤖 Strategy Assistant")
    groq_api_key = st.secrets.get("GROQ_API_KEY")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_box = st.container(height=650, border=True)

    with chat_box:
        for m in st.session_state.messages:
            avatar = "🧬" if m["role"] == "assistant" else "👤"
            with st.chat_message(m["role"], avatar=avatar):
                st.markdown(m["content"])

    if prompt := st.chat_input("Analyze HCP opportunity..."):
        with chat_box:
            st.chat_message("user", avatar="👤").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Routing..."):
            intent = get_intent(prompt, groq_api_key)
        
        with chat_box:
            with st.chat_message("assistant", avatar="🧬"):
                if intent == "OPPORTUNITY":
                    scorecard = get_hcp_scorecard(prompt, df)
                    if scorecard:
                        # UI Update: NPI is the Header
                        st.markdown(f"### 🎯 NPI: {scorecard['npi']}")
                        st.success(f"**State:** {scorecard['state']} | **Type:** {scorecard['type']}")
                        
                        st.markdown(f"""
                        **Opportunity Scorecard:**
                        - **Propensity Score:** {scorecard['score']:.1%}
                        - **Medicare Payment Volume:** ${scorecard['payment']:,.0f}
                        - **Model Drivers:** {scorecard['drivers']}
                        """)
                        
                        # --- ENHANCED AI INSIGHT ---
                        try:
                            client = Groq(api_key=groq_api_key)
                            persona = SYSTEM_PERSONAS.get("data_analyst", "You are a Merck Lead Data Scientist.")
                            
                            analysis_prompt = f"Explain the strategic importance of this HCP target based on these drivers: {scorecard['drivers']}. Keep it to 2 concise sentences."
                            
                            res = client.chat.completions.create(
                                model="llama3-8b-8192",
                                messages=[
                                    {"role": "system", "content": persona},
                                    {"role": "user", "content": analysis_prompt}
                                ],
                                temperature=0.3
                            )
                            st.info(f"💡 **AI Insight:** {res.choices[0].message.content}")
                        except Exception as e:
                            st.warning("⚠️ High-level drivers identified, but AI summary was unavailable.")
                    else:
                        st.error("No matches found for that specific state.")
                
                elif intent == "NEWS":
                    st.info("📰 Scanning external market intelligence for competitor updates...")
                else:
                    st.write(f"Intent {intent} identified. Processing strategy...")

        st.session_state.messages.append({"role": "assistant", "content": f"Routed as: {intent}"})