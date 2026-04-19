import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("💬 Merck Chatbot")
st.write(
    "This is a fast, free-tier chatbot powered by Groq. "
    "You can get a free API key at [console.groq.com](https://console.groq.com/)."
)

# Ask user for their Groq API key via `st.text_input`.
# TIP: For your final version, you can move this to "Secrets" in the Streamlit dashboard.
groq_api_key = st.text_input("Groq API Key", type="password")

if not groq_api_key:
    st.info("Please add your Groq API key to continue.", icon="🗝️")
else:

    # 1. CHANGE: Initialize the client to point to Groq's URL
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
    if prompt := st.chat_input("What is up?"):

        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. CHANGE: Use a free Groq model name 
        # Llama 3.1 8B is typically the best for fast, free testing
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
            st.session_state.messages.append({"role": "assistant", "content": response})