import streamlit as st
from groq import Groq
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Mir MUHAMMAD & Co | AI",
    page_icon="🚀",
    layout="centered"
)

# --- CUSTOM STYLING (Ensuring no black background) ---
st.markdown("""
    <style>
        .stApp {
            background-color: #ffffff;
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f8f9fa;
            color: #333;
            text-align: center;
            padding: 10px;
            font-weight: bold;
            border-top: 1px solid #e7e7e7;
        }
    </style>
    <div class="footer">
        Developed By Mir MUHAMMAD Rafique And Co
    </div>
    """, unsafe_allow_index=True)

# --- GROQ API SETUP ---
# It's best to set your key in your system environment or a .env file
# For quick testing, you can replace 'your_api_key_here' directly
api_key = os.environ.get("GROQ_API_KEY") or "your_api_key_here"
client = Groq(api_key=api_key)

# --- UI HEADER ---
st.title("Groq AI Assistant")
st.write("Enter your prompt below to interact with the LLM.")

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        try:
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=False,
            )
            full_response = response.choices[0].message.content
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Error: {e}")
