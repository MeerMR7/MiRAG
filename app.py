import os
import streamlit as st
import pdfplumber
import requests
import datetime

# -------------------- 1. CONFIGURATION --------------------
st.set_page_config(page_title="Academic Policy Assistant", layout="centered", page_icon="📜")

# Securely get your API Key from Streamlit Secrets
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    st.error("⚠️ GROQ_API_KEY not found! Please add it to your Secrets.")
    st.stop()

# --- THE AUTOMATIC PATH ---
# Ensure this file is in the same folder as this script
PDF_PATH = "Academic-Policy-Manual-for-Students2.pdf" 
MODEL_NAME = "llama-3.1-8b-instant"

# -------------------- 2. AUTOMATIC PDF LOADING --------------------
@st.cache_data(show_spinner="Reading Academic Policy Manual...")
def load_policy_data(path: str, max_chars: int = 600):
    if not os.path.exists(path):
        st.error(f"❌ File not found at: {path}. Please place the PDF in the folder.")
        return []
    
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            tx = page.extract_text()
            if tx:
                text += tx + "\n"

    # Split into chunks for retrieval
    raw_parts = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    buf = ""

    for part in raw_parts:
        if len(buf) + len(part) <= max_chars:
            buf += " " + part
        else:
            if buf.strip():
                chunks.append(buf.strip())
            buf = part

    if buf.strip():
        chunks.append(buf.strip())
    return chunks

# This runs automatically when the app starts
pdf_chunks = load_policy_data(PDF_PATH)

# -------------------- 3. RETRIEVAL ENGINE --------------------
def find_relevant_context(query: str, top_k: int = 3):
    if not pdf_chunks:
        return ""
        
    q_words = set(query.lower().split())
    scored = []

    for ch in pdf_chunks:
        ch_words = set(ch.lower().split())
        score = len(q_words & ch_words)
        if score > 0:
            scored.append((score, ch))

    if not scored:
        return ""

    scored.sort(reverse=True, key=lambda x: x[0])
    return "\n\n".join([c for _, c in scored[:top_k]])

# -------------------- 4. API & LOGIC --------------------
def get_ai_response(question: str, history):
    context = find_relevant_context(question)
    today = datetime.datetime.now().strftime("%d %B %Y")
    
    # System Prompt focused strictly on Academic Policy
    if not context.strip():
        system_prompt = f"You are an Academic Policy Assistant. Today is {today}. Provide clear, direct information. Do not mention searching."
    else:
        system_prompt = f"""
You are an Academic Policy Assistant. Use the PDF Context below as your primary source. 
If the answer isn't there, use your general knowledge. Today is {today}.

Rules:
- Give professional and direct answers.
- Never say "I am checking the PDF" or "I am searching".
- Do not limit knowledge to 2023.

PDF Context:
{context}
"""

    messages = [{"role": "system", "content": system_prompt}]
    for m in history[-6:]:
        messages.append(m)
    messages.append({"role": "user", "content": question})

    # API Call
    payload = {"model": MODEL_NAME, "messages": messages, "temperature": 0.3}
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
        return response.json()["choices"][0]["message"]["content"]
    except:
        return "⚠️ I encountered an error connecting to the policy database."

# -------------------- 5. UI --------------------
st.title("📜 Academic Policy Assistant")
st.info(f"Connected to: `{PDF_PATH}`")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Assalam o Alaikum! I am your Academic Policy Assistant. How can I help you today?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask a question about university policy..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Reviewing policy..."):
            answer = get_ai_response(user_input, st.session_state.messages[:-1])
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
