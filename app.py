import streamlit as st
import pdfplumber
import requests
import datetime
import os
import re

# -------------------- 1. PAGE CONFIG & BRANDING --------------------
st.set_page_config(page_title="Mir MUHAMMAD Rafique & Co Advisor", layout="centered")

# Professional Light Theme CSS
st.markdown("""
    <style>
    .stApp { background-color: #f8fafc !important; color: #0f172a !important; }
    [data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e2e8f0; }
    .footer {
        position: fixed; left: 0; bottom: 0; width: 100%;
        background-color: #ffffff; color: #1e293b;
        text-align: center; padding: 10px; font-weight: bold;
        border-top: 1px solid #e2e8f0; z-index: 100;
    }
    </style>
    <div class='footer'>Developed By Mir MUHAMMAD Rafique And Co</div>
    """, unsafe_allow_html=True)

# -------------------- 2. SETUP & PATHS --------------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
# Updated to your specific PDF filename
PDF_PATH = "Academic-Policy-Manual-for-Students2.pdf"
MODEL_NAME = "llama-3.1-70b-versatile" # Higher intelligence for production

# -------------------- 3. LOAD + CHUNK PDF --------------------
@st.cache_data
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

# -------------------- 4. SIMPLE RETRIEVAL --------------------
def retrieve_context(query: str, top_k: int = 4):
    if not pdf_chunks:
        return ""
    q_words = set(re.findall(r'\w+', query.lower()))
    scored = []

    for ch in pdf_chunks:
        ch_words = set(re.findall(r'\w+', ch.lower()))
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
        "temperature": 0.3, # Lower temperature = more professional/accurate
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ System Connection Error: {str(e)}"

# -------------------- 6. RAG LOGIC --------------------
def get_answer(question: str, history):
    context = retrieve_context(question)
    today = datetime.datetime.now().strftime("%d %B %Y")
    
    # Check if we have PDF data to work with
    if len(context.strip()) < 40:
        system_prompt = f"""
You are the Mir MUHAMMAD Rafique And Co AI Advisor.
Today's Date: {today}

Rules:
- Provide clear, direct, and professional answers.
- Use your internal updated knowledge.
- Never say "I am searching" or "Checking the files". 
- Speak with authority and confidence.
"""
    else:
        system_prompt = f"""
You are the Mir MUHAMMAD Rafique And Co AI Advisor.
Today's Date: {today}

Primary Source (Academic Policy Manual):
---------------------
{context}
---------------------

Rules:
- Base your answers primarily on the Academic Policy Manual context provided above.
- Provide confident and direct advice.
- If info is missing, guide the user to the official Administration Office.
- Do NOT use phrases like "Based on the text provided" or "I am researching".
"""

    messages = [{"role": "system", "content": system_prompt}]
    for m in history[-5:]: # Maintain short memory for better context
        messages.append(m)
    messages.append({"role": "user", "content": question})

    return llama_chat(messages)

# -------------------- 7. STREAMLIT UI --------------------
st.title("📋 Academic Policy Advisor")
st.subheader("Mir MUHAMMAD Rafique And Co | Student Portal")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hello. I am the AI Advisor for Mir MUHAMMAD Rafique And Co. How can I assist you with academic policies today?"}
    ]

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_input := st.chat_input("Enter your policy question here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Processing inquiry..."):
            answer = get_answer(user_input, st.session_state.messages)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
