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

# --- 3. LAYOUT DEFINITION (BI on Left, AI on Right) ---
# We use a 2-column layout to keep the map and chat side-by-side
main_col, chat_col = st.columns([2.2, 1], gap="medium")

# --- LEFT COLUMN: BI & VISUALIZATIONS ---
with main_col:
    # Header & Credits
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

    # BI Reporting Layer (KPIs)
    st.markdown("### 📊 Market Intelligence Overview")
    total_providers = len(df)
    predicted_yes = len(df[df['pred_class'] == 1])
    unique_zips = df['Rndrng_Prvdr_Zip5'].nunique()
    total_types = df['Cleaned_Prvdr_Type'].nunique()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Providers", f"{total_providers:,}")
    kpi2.metric("High Propensity (AI Yes)", f"{predicted_yes:,}")
    kpi3.metric("Unique Zip5s", f"{unique_zips:,}")
    kpi4.metric("Specialties", f"{total_types:,}")

    # Geographic Heatmap (Now fixed in position)
    st.plotly_chart(plot_executive_map(df), use_container_width=True)

# --- RIGHT COLUMN: AI CONVERSATIONAL INTERFACE ---
with chat_col:
    st.markdown("### 🤖 Strategy Assistant")
    
    # Check for API Key
    groq_api_key = st.secrets.get("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Please add your GROQ_API_KEY to Streamlit Secrets.")
        st.stop()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Scrollable container for chat history (Fixed height keeps UI stable)
    chat_box = st.container(height=650, border=True)

    with chat_box:
        for message in st.session_state.messages:
            # Using standard emojis to ensure 100% stability across all Streamlit versions
            avatar = "🧬" if message["role"] == "assistant" else "👤"
            
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

    # Input area for the Chat
    if prompt := st.chat_input("Analyze HCP opportunity..."):
        # 1. Display user message in the UI
        with chat_box:
            st.chat_message("user", avatar="👤").markdown(prompt)
        
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 2. Process Intent
        with st.spinner("Routing intent..."):
            try:
                intent = get_intent(prompt, groq_api_key)
            except Exception as e:
                st.error(f"Error calling Router: {e}")
                intent = "NEWS" # Fallback

        # 3. Display Assistant Response in the UI
        with chat_box:
            with st.chat_message("assistant", avatar="🧬"):
                if intent == "OPPORTUNITY":
                    st.markdown("### 🎯 Target Identified")
                    st.success("The AI model suggests a high-propensity provider opportunity.")
                    st.markdown("---")
                    st.info("💡 **Insight:** This HCP ranks in the top 5% for Medicare oncology volume. I can generate a detailed Opportunity Scorecard for this target.")
                
                elif intent == "MARKETING":
                    st.markdown("### 📈 Marketing Strategy")
                    st.warning("Analyzing Omnichannel engagement optimization paths...")
                    st.markdown("Determining the best frequency for field rep visits and digital touchpoints based on HCP historical response.")
                
                elif intent == "NEWS":
                    st.markdown("### 📰 Market Intelligence")
                    st.info("Scanning competitive IO intelligence and FDA pipelines...")
                    st.markdown("Retrieving latest clinical trial data for IO combinations and biosimilar entry dates.")

        # Save assistant message to session state
        st.session_state.messages.append({"role": "assistant", "content": f"System Routed to: {intent}"})