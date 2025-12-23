import os
import streamlit as st
import pdfplumber
import requests
import datetime

# -------------------- 1. APP CONFIGURATION --------------------
st.set_page_config(page_title="Academic Policy Assistant", layout="centered", page_icon="📜")

# Securely get your API Key from Streamlit Secrets
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    st.error("⚠️ Error: 'GROQ_API_KEY' not found in Secrets. Please add it to your .streamlit/secrets.toml file.")
    st.stop()

# HARDCODED FILE PATH
PDF_PATH = "Academic-Policy-Manual-for-Students2.pdf" 
MODEL_NAME = "llama-3.1-8b-instant"

# -------------------- 2. AUTOMATIC PDF PROCESSING --------------------
@st.cache_data(show_spinner="Initializing Academic Policy Manual...")
def load_policy_manual(path: str, max_chars: int = 600):
    if not os.path.exists(path):
        st.error(f"❌ File Not Found: Please place '{path}' in the same folder as this script.")
        return []
    
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return []

    # Chunking for retrieval
    raw_lines = [line.strip() for line in text.split("\n") if line.strip()]
    chunks = []
    buffer = ""

    for line in raw_lines:
        if len(buffer) + len(line) <= max_chars:
            buffer += " " + line
        else:
            if buffer.strip():
                chunks.append(buffer.strip())
            buffer = line

    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks

# Auto-run the loader
policy_chunks = load_policy_manual(PDF_PATH)

# -------------------- 3. RETRIEVAL ENGINE --------------------
def get_relevant_policy_text(query: str, top_k: int = 3):
    if not policy_chunks:
        return ""
        
    query_words = set(query.lower().split())
    scored_chunks = []

    for chunk in policy_chunks:
        chunk_words = set(chunk.lower().split())
        # Score based on keyword overlap
        score = len(query_words & chunk_words)
        if score > 0:
            scored_chunks.append((score, chunk))

    if not scored_chunks:
        return ""

    # Sort by relevance and return top matches
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    return "\n\n".join([item[1] for item in scored_chunks[:top_k]])

# -------------------- 4. GROQ API COMMUNICATION --------------------
def query_policy_ai(question: str, history):
    context = get_relevant_policy_text(question)
    current_date = datetime.datetime.now().strftime("%d %B %Y")
    
    # Professional System Prompt
    if not context.strip():
        system_prompt = f"You are a professional Academic Policy Assistant. Today is {current_date}. Answer questions based on general university standards. Do not say you are searching."
    else:
        system_prompt = f"""
You are an Academic Policy Assistant. Use the following PDF context to answer the user. 
Today's date is {current_date}.

Rules:
- Be precise, formal, and direct.
- Do NOT mention "searching the PDF" or "checking the manual."
- Use the context provided below as your primary source.

PDF Context:
{context}
"""

    messages = [{"role": "system", "content": system_prompt}]
    # Add history for continuity
    for m in history[-6:]:
        messages.append(m)
    messages.append({"role": "user", "content": question})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.2, # Lower temperature for higher accuracy
    }
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ System Error: Unable to reach the policy engine. ({str(e)})"

# -------------------- 5. USER INTERFACE --------------------
st.title("📜 Academic Policy Assistant")
st.caption(f"Currently indexing: {PDF_PATH}")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome. I have indexed the Academic Policy Manual. How may I assist you with university regulations today?"}
    ]

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if user_query := st.chat_input("Ask about attendance, grading, probation, etc..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Consulting manual..."):
            ai_response = query_policy_ai(user_query, st.session_state.messages[:-1])
        st.markdown(ai_response)
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
