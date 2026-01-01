import os
import streamlit as st
import requests
import datetime

# -------------------- 1. BRANDING & CONFIG --------------------
st.set_page_config(page_title="MiRAG | Chat", layout="centered", page_icon="💬")

# Securely fetch API Key from Streamlit Secrets
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "llama-3.1-8b-instant"

# -------------------- 2. CHAT INTELLIGENCE --------------------
def mirag_chat(question, history):
    today = datetime.datetime.now().strftime("%d %B %Y")
    
    # This prompt tells the AI who created it and how to behave
    system_prompt = f"""
    You are MiRAG, a friendly and helpful AI assistant created by 
    Mir MUHAMMAD Rafique and Hasnain Ali Raza. 
    Current Date: {today}. 
    Your goal is to have a natural conversation, answer questions, and be polite.
    """

    # Combine system prompt with conversation history
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": question}]

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions", 
            json={"model": MODEL_NAME, "messages": messages, "temperature": 0.7},
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"}
        )
        return response.json()["choices"][0]["message"]["content"]
    except:
        return "⚠️ I'm having trouble connecting right now. Please check your API key."

# -------------------- 3. USER INTERFACE --------------------
st.title("💬 MiRAG Chat")
st.subheader("Developed by Mir MUHAMMAD Rafique & Hasnain Ali Raza")

# Sidebar for extra info
with st.sidebar:
    st.title("Bot Info")
    st.write("This is a general-purpose chatbot for open conversation.")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Assalam o Alaikum! I am MiRAG. Let's chat!"}]

# Display the conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle User Input
if user_input := st.chat_input("Say something..."):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("MiRAG is typing..."):
            ans = mirag_chat(user_input, st.session_state.messages[:-1])
            st.markdown(ans)
    
    # Save assistant response to state
    st.session_state.messages.append({"role": "assistant", "content": ans})
