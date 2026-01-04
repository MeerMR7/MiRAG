import streamlit as st
import pdfplumber
import requests
import datetime
import os

# -------------------- 1. PAGE CONFIG & BRANDING --------------------
st.set_page_config(page_title="MiRAG | Academic Advisor", layout="centered", page_icon="🤖")

# Custom CSS for Light Theme and Professional Footer
st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f8fafc;
        color: #334155;
        text-align: center;
        padding: 10px;
        font-weight: bold;
        border-top: 1px solid #e2e8f0;
        z-index: 100;
    }
    /* Style for Assistant Messages */
    .stChatMessage {
        background-color: #f1f5f9;
        border-radius: 10px;
    }
    </style>
    <div class='footer'>Developed By Mir MUHAMMAD Rafique And Co</div>
    """, unsafe_allow_html=True)

# -------------------- 2. SETUP & SECRETS --------------------
# Ensure GROQ_API_KEY is added to Streamlit Cloud Secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
PDF_PATH = "Academic-Policy-Manual-for-Students2.pdf"
MODEL_NAME = "llama-3.1-8b-instant"

# -------------------- 3. LOAD + CHUNK PDF --------------------
@st.cache_data
def load_chunks(max_chars: int = 600):
    if not os.path.exists(PDF_PATH):
        return []
    
    text = ""
    try:
        with pdfplumber.open(PDF_PATH) as pdf:
            for page in pdf.pages:
                tx = page.extract_text()
                if tx:
                    text += tx + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return []

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

# -------------------- 4. SIMPLE RETRIEVAL --------------------
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

# -------------------- 5. GROQ API CALL --------------------
def llama_chat(messages):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.4,
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ Groq API Error: {str(e)}"

# -------------------- 6. RAG + SYSTEM PROMPT --------------------
def get_answer(question: str, history):
    context = retrieve_context(question)
    today = datetime.datetime.now().strftime("%d %B %Y")
    pdf_strength = len(context.strip())

    if pdf_strength < 50:
        system_prompt = f"""
You are MiRAG, the official AI Assistant of Mir MUHAMMAD Rafique And Co.
Rules:
- Give clear and direct answers.
- Use your updated general knowledge (today = {today}).
- Do NOT say anything about "searching" or "not knowing".
- Never restrict information to the year 2023.
"""
    else:
        system_prompt = f"""
You are MiRAG, the official AI Assistant of Mir MUHAMMAD Rafique And Co.
Use the following PDF text as your main reference. 

PDF Context:
---------------------
{context}
---------------------

Rules:
- Provide confident and direct professional answers based on the context.
- If updated information (today = {today}) is needed, include it naturally.
- Do NOT say "I am searching" or "Checking files".
"""

    messages = [{"role": "system", "content": system_prompt}]
    for m in history[-6:]:
        messages.append(m)
    messages.append({"role": "user", "content": question})

    return llama_chat(messages)

# -------------------- 7. STREAMLIT UI --------------------
st.title("🤖 MiRAG")
st.write("### Academic Policy Expert")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Assalam o Alaikum! 👋 I am MiRAG. I am here to help you with Academic Policies from the manual. How can I assist you today?"}
    ]

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_input := st.chat_input("Apna sawal likho..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("MiRAG is thinking..."):
            answer = get_answer(user_input, st.session_state.messages)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
