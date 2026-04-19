import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("💬 Merck Chatbot")
st.write(
    "This is a fast, professional chatbot powered by Groq and Llama 3.1."
)

# 1. NEW SECRETS LOGIC: 
# This pulls the key from your Streamlit Cloud "Secrets" settings automatically.
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("Missing secret: 'GROQ_API_KEY'. Please add it to your Streamlit Cloud settings.")
    st.stop()

# 2. Initialize the client using the secret key
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key
)

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("How can I help you today?"):

    # Store and display the current prompt.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Generate response using Groq's Llama 3.1 8B model
    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant", 
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ],
        stream=True,
    )
    
    with st.chat_message("assistant"):
        response = st.write_stream(stream)
    
    # Save the assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})