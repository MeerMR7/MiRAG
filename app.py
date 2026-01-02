import os
import streamlit as st
import pdfplumber
import requests
import datetime

# -------------------- BASIC SETUP --------------------
st.set_page_config(page_title="MiRAG | Academic Assistant", layout="centered", page_icon="🎓")

# API Key Handling
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- UPDATED PATH ---
# Ensure this file is in your main container/folder
PDF_PATH = "Academic-Policy-Manual-for-Students2.pdf" 
MODEL_NAME = "llama-3.1-8b-instant"

# -------------------- LOAD + CHUNK PDF --------------------
@st.cache_data(show_spinner="Indexing Academic Policy Manual...")
def load_chunks(max_chars: int = 700):
    if not os.path.exists(PDF_PATH):
        return None
    
    text = ""
    try:
        with pdfplumber.open(PDF_PATH) as pdf:
            for page in pdf.pages:
                tx = page.extract_text()
                if tx:
                    text += tx + "\n"

        # Break text into sections for the AI to read easily
        raw_parts = [p.strip() for p in text.split("\n") if p.strip()]
        chunks = []
        buf = ""

        for part in raw_parts:
            if len(buf) + len(part) <= max_chars:
                buf += " " + part
            else:
                chunks.append(buf.strip())
                buf = part

        if buf:
            chunks.append(buf.strip())
        return chunks
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return None

pdf_chunks = load_chunks()

# -------------------- SIMPLE RETRIEVAL --------------------
def retrieve_context(query: str, top_k: int = 3):
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

# -------------------- GROQ API CALL --------------------
def llama_chat(messages):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.2, # Lower temperature for factual accuracy
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ Groq API Error: {str(e)}"

# -------------------- RAG LOGIC --------------------
def get_answer(question: str, history):
    context = retrieve_context(question)
    today = datetime.datetime.now().strftime("%d %B %Y")
    
    # Custom system prompt for your specific manual
    system_prompt = f"""
You are MiRAG, an Academic Assistant for Iqra University.
Created by: Mir MUHAMMAD Rafique and Hasnain Ali Raza.

Use the provided PDF context to answer university-related questions accurately.
If the information is not in the manual, answer based on general academic standards but mention it's general info.

Current Date: {today}

ACADEMIC POLICY CONTEXT:
---------------------
{context}
---------------------

Rules:
- Be professional and helpful.
- For grading, attendance, or admissions, prioritize the PDF context.
"""

    messages = [{"role": "system", "content": system_prompt}]
    for m in history[-6:]:
        messages.append(m)
    messages.append({"role": "user", "content": question})

    return llama_chat(messages)

# -------------------- STREAMLIT UI --------------------
st.title("🎓 MiRAG: Academic Assistant")
st.subheader("Developed by Mir MUHAMMAD Rafique & Hasnain Ali Raza")

# Status Check in Sidebar
with st.sidebar:
    if pdf_chunks:
        st.success(f"✅ Manual Loaded: {len(pdf_chunks)} sections found.")
    else:
        st.error("❌ Policy Manual Not Found")
        st.info(f"Make sure '{PDF_PATH}' is in the main project folder.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Assalam o Alaikum! I am MiRAG. I can help you with grading, admissions, and attendance rules from the Policy Manual."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about university policies...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Consulting Policy Manual..."):
            answer = get_answer(user_input, st.session_state.messages)
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
