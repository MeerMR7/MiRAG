import os
import streamlit as st
import pdfplumber
import requests
import datetime

# -------------------- 1. IDENTITY & CONFIG --------------------
st.set_page_config(page_title="HasMir | Academic Assistant", layout="centered", page_icon="🤖")

# Securely fetch API Key from Streamlit Secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# --- GITHUB CONNECTED PATH ---
# Looks for the file in the same main folder on GitHub
PDF_PATH = "Academic-Policy-Manual-for-Students2.pdf"
MODEL_NAME = "llama-3.1-8b-instant"


# -------------------- 2. LOAD + CHUNK PDF --------------------
@st.cache_data(show_spinner="HasMir is reading the Policy Manual...")
def load_chunks(max_chars: int = 700):
    if not os.path.exists(PDF_PATH):
        return None
        
    text = ""
    with pdfplumber.open(PDF_PATH) as pdf:
        for page in pdf.pages:
            tx = page.extract_text()
            if tx:
                text += tx + "\n"

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

pdf_chunks = load_chunks()


# -------------------- 3. RETRIEVAL LOGIC --------------------
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


# -------------------- 4. GROQ API CALL --------------------
def llama_chat(messages):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.2, # Lower temperature for better factual accuracy
    }

    response = requests.post(url, json=payload, headers=headers)
    result = response.json()

    try:
        return result["choices"][0]["message"]["content"]
    except:
        return "⚠️ Error connecting to HasMir's brain. Please check the API Key."


# -------------------- 5. BRAIN (RAG LOGIC) --------------------
def get_answer(question: str, history):
    context = retrieve_context(question)
    today = datetime.datetime.now().strftime("%d %B %Y")
    
    # System Prompt with Developer Credits
    system_prompt = f"""
You are HasMir, a professional Academic AI Assistant.
Developed by: Mir MUHAMMAD and Hasnain Ali Raza.

Rules:
- Provide clear, direct, and helpful answers in English.
- Use the PDF Context below as your main source for academic rules.
- If information isn't in the PDF, use your general knowledge (Current Date: {today}).
- Do NOT say "I am searching" or "I am looking at the PDF". Just answer.

PDF Context:
---------------------
{context}
---------------------
"""

    messages = [{"role": "system", "content": system_prompt}]
    for m in history[-6:]:
        messages.append(m)
    messages.append({"role": "user", "content": question})

    return llama_chat(messages)


# -------------------- 6. STREAMLIT UI --------------------
st.title("🤖 HasMir's ChatBot")
st.caption("Official Academic Assistant | Developed by Mir MUHAMMAD & Hasnain Ali Raza")

# Sidebar Status
with st.sidebar:
    st.header("System Status")
    if pdf_chunks:
        st.success(f"✅ Linked to Policy Manual")
    else:
        st.error("❌ PDF Manual Not Found")
        st.info("Check if 'Academic-Policy-Manual-for-Students2.pdf' is in the GitHub folder.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hello! I am HasMir. I have been developed by Mir MUHAMMAD and Hasnain Ali Raza to assist you with academic policies. How can I help you today?"}
    ]

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask me about grading, attendance, or admissions...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("HasMir is thinking..."):
            answer = get_answer(user_input, st.session_state.messages[:-1])
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
