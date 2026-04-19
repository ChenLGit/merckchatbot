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

# --- 1. SETUP PAGE CONFIG & TITLE ---
st.set_page_config(page_title="Merck Data Science Hub", layout="wide")

# Custom two-line centered title
st.markdown("""
    <div style="text-align: center;">
        <h1 style="margin-bottom: 0px;">Merck Keytruda</h1>
        <h2 style="margin-top: 0px; font-weight: normal; color: #555;">Provider Targeting Strategy AI Application</h2>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- 2. DATA LOADING (Path: data/raw/) ---
@st.cache_data
def load_data():
    # Points exactly to your new data folder structure
    data_path = os.path.join(current_dir, 'data', 'raw', 'MerckAI_table.csv')
    
    if not os.path.exists(data_path):
        st.error(f"Data file not found at: {data_path}")
        st.stop()
        
    df = pd.read_csv(data_path)
    return df

df = load_data()

# --- 3. BI REPORTING LAYER (KPIs) ---
st.markdown("### 📊 Market Intelligence Overview")
total_providers = len(df)
predicted_yes = len(df[df['pred_class'] == 1])
unique_zips = df['Rndrng_Prvdr_Zip5'].nunique()
total_types = df['Cleaned_Prvdr_Type'].nunique()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Providers", f"{total_providers:,}")
kpi2.metric("High Propensity (AI Yes)", f"{predicted_yes:,}")
kpi3.metric("Total Unique Zip5 Areas", f"{unique_zips:,}")
kpi4.metric("Provider Specialties Covered", f"{total_types:,}")

# --- 4. GEOGRAPHIC HEATMAP ---
st.plotly_chart(plot_executive_map(df), use_container_width=True)

st.markdown("---")

# --- 5. GROQ & CHAT INTERFACE ---
groq_api_key = st.secrets.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("Please add your GROQ_API_KEY to Streamlit Secrets.")
    st.stop()

# Chat interface setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if prompt := st.chat_input("Ask about Merck providers, marketing, or competitors..."):
    
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Analyzing intent..."):
        intent = get_intent(prompt, groq_api_key)
#Output 
    with st.chat_message("assistant"):
        if intent == "OPPORTUNITY":
            st.success("🎯 **Intent Detected: Provider Opportunity Analysis**")
            st.markdown("The AI model identifies these providers based on clinical volume and diagnostic patterns.")
            st.info("💡 *Tip: Try asking 'Who are the top 5 oncology targets in Texas?'*")
            
        elif intent == "MARKETING":
            st.warning("📈 **Intent Detected: Marketing Strategy**")
            st.markdown("Routing to the Omnichannel engine for HCP engagement optimization...")
            
        elif intent == "NEWS":
            st.info("📰 **Intent Detected: Market Intelligence**")
            st.markdown("Scanning latest competitive intelligence for Keytruda biosimilars and IO competitors...")
            
    st.session_state.messages.append({"role": "assistant", "content": f"System Routed to: {intent}"})