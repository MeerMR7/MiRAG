import streamlit as st
import pdfplumber
import requests
import datetime

# -------------------- BASIC SETUP --------------------
st.set_page_config(page_title="Mir MUHAMMAD Rafique's ChatBot", layout="centered")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
PDF_PATH = "Academic-Policy-Manual-for-Students2.pdf"
MODEL_NAME = "llama-3.1-8b-instant"


# -------------------- LOAD + CHUNK PDF --------------------
@st.cache_data
def load_chunks(max_chars: int = 600):
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


# -------------------- SIMPLE RETRIEVAL --------------------
def retrieve_context(query: str, top_k: int = 3):
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
        "temperature": 0.4,
    }

    response = requests.post(url, json=payload, headers=headers)
    result = response.json()

    try:
        return result["choices"][0]["message"]["content"]
    except:
        return "⚠️ Groq API Error:\n" + str(result)


# -------------------- RAG + UPDATED INFO (NO SEARCHING TEXT) --------------------
def get_answer(question: str, history):
    context = retrieve_context(question)
    today = datetime.datetime.now().strftime("%d %B %Y (%Y)")
    pdf_strength = len(context.strip())

    if pdf_strength < 50:
        # PDF does not contain relevant information → use AI updated knowledge
        system_prompt = f"""
You are Mir MUHAMMAD Rafique's Chatbot.

Rules:
- Give clear and direct answers.
- Use your updated general knowledge (today = {today}).
- Do NOT say anything about "searching", "checking", "researching", or "not knowing".
- Never restrict information to the year 2023.
"""
    else:
        # PDF has useful context → use it first, but allow updated info too
        system_prompt = f"""
You are Mir MUHAMMAD Rafique's Chatbot.

Use the following PDF text as your main reference. 
If updated information (today = {today}) is needed, include it naturally.

PDF Context:
---------------------
{context}
---------------------

Rules:
- Provide confident and direct answers.
- Do NOT say "I am searching" or "I am researching".
- Never limit your knowledge to only 2023.
"""

    # Build message list
    messages = [{"role": "system", "content": system_prompt}]
    
    for m in history[-6:]:
        messages.append(m)

    messages.append({"role": "user", "content": question})

    return llama_chat(messages)


# -------------------- STREAMLIT UI --------------------
st.title("🤖 Mir MUHAMMAD Rafique's Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Assalam o Alaikum! 👋I Am  Mir MUHAMMAD Rafique's Chatbot. "
                    "Ask Me About Academic Policies"}
    ]

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Apna sawal likho...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Soch raha hoon..."):
            answer = get_answer(user_input, st.session_state.messages)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
