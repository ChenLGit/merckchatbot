import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')

if src_path not in sys.path:
    sys.path.append(src_path)

import streamlit as st
import pandas as pd
from groq import Groq

from src.router import get_intent
from src.prompts import SYSTEM_PROMPTS

# 1. Setup Page Config & Title
st.set_page_config(page_title="Merck Data Science Hub", layout="wide")
st.title("🧬 Merck AI Strategy Dashboard")

# 2. Retrieve Groq API Key from Secrets
# Note: Ensure you have "GROQ_API_KEY" in your Streamlit Cloud Secrets
groq_api_key = st.secrets.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("Please add your GROQ_API_KEY to Streamlit Secrets.")
    st.stop()

# 3. Chat Interface Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Handle User Input
if prompt := st.chat_input("Ask about Merck providers, marketing, or competitors..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Execute Routing Logic
    with st.spinner("Analyzing intent..."):
        intent = get_intent(prompt, groq_api_key)

    # Generate Response based on Intent
    with st.chat_message("assistant"):
        if intent == "OPPORTUNITY":
            st.success("🎯 **Intent Detected: Provider Opportunity Analysis**")
            st.markdown("Routing to the Data Science engine to analyze SHAP values and doctor rankings...")
            # TODO: Add call to src/visualizations.py
            
        elif intent == "MARKETING":
            st.warning("📈 **Intent Detected: Marketing Strategy**")
            st.markdown("Routing to the Marketing Strategist engine for channel optimization tips...")
            # TODO: Add call to src/rag_engine.py
            
        elif intent == "NEWS":
            st.info("📰 **Intent Detected: Market Intelligence**")
            st.markdown("Searching for the latest news on Merck and Keytruda competitors...")
            # TODO: Add call to search tools
            
    # Add assistant response to history (simple placeholder for now)
    st.session_state.messages.append({"role": "assistant", "content": f"Routed to {intent}"})