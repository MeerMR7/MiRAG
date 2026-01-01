import os
import streamlit as st
import pdfplumber
import requests
import datetime

# -------------------- 1. BRANDING & CONFIG --------------------
st.set_page_config(page_title="MiRAG | Academic Assistant", layout="centered", page_icon="🤖")

if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    st.error("⚠️ GROQ_API_KEY missing in Secrets!")
    st.stop()

# AUTO-FILE PATH
PDF_PATH = "Academic-Policy-Manual-for-Students2.pdf" 
MODEL_NAME = "llama-3.1-8b-instant"

# -------------------- 2. AUTO-LOAD POLICY --------------------
@st.cache_data(show_spinner="MiRAG is syncing with the Policy Manual...")
def load_mirag_brain(path: str):
    if not os.path.exists(path):
        return []
    
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            tx = page.extract_text()
            if tx: text += tx + "\n"

    # Precise chunking for better retrieval
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    chunks, buf = [], ""
    for line in lines:
        if len(buf) + len(line) <= 600:
            buf += " " + line
        else:
            chunks.append(buf.strip())
            buf = line
    if buf: chunks.append(buf.strip())
    return chunks

policy_data = load_mirag_brain(PDF_PATH)

# -------------------- 3. SMART RETRIEVAL --------------------
def get_context(query: str):
    if not policy_data: return ""
    q_words = set(query.lower().split())
    scored = sorted([(len(q_words & set(c.lower().split())), c) for c in policy_data], reverse=True, key=lambda x: x[0])
    return "\n\n".join([c for score, c in scored[:3] if score > 0])

# -------------------- 4. MiRAG INTELLIGENCE --------------------
def mirag_chat(question: str, history):
    context = get_context(question)
    today = datetime.datetime.now().strftime("%d %B %Y")
    
    # Updated 2025 Intelligence (Grading/Attendance updates)
    system_prompt = f"""
You are MiRAG (Mir MUHAMMAD Rafique's Chat Bot), a professional Academic Policy Assistant.
Current Date: {today}.

CORE INSTRUCTIONS:
1. Primary Source: Use the PDF context provided.
2. 2025 Updates: Be aware that for 2025, the passing threshold is now 50% for undergraduates. 
   A new grade "XF" is used for failure due to attendance shortage. 
   The Probation limit is now 1.70 Semester GPA.
3. Style: Professional, confident, and direct. 
4. DO NOT mention "searching the PDF" or "checking my database".

PDF Context:
{context}
"""

    messages = [{"role": "system", "content": system_prompt}]
    for m in history[-6:]: messages.append(m)
    messages.append({"role": "user", "content": question})

    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", 
                          json={"model": MODEL_NAME, "messages": messages, "temperature": 0.3},
                          headers={"Authorization": f"Bearer {GROQ_API_KEY}"})
        return r.json()["choices"][0]["message"]["content"]
    except:
        return "⚠️ MiRAG is currently experiencing a connection issue."

# -------------------- 5. USER INTERFACE --------------------
st.title("🤖 MiRAG")
st.subheader("Mir MUHAMMAD Rafique & Hasnain Ali Raza's Chat Bot")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Assalam o Alaikum! I am MiRAG. I have indexed your Academic Policy. How can I assist you?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if user_input := st.chat_input("Ask MiRAG about university policies..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("MiRAG is analyzing..."):
            ans = mirag_chat(user_input, st.session_state.messages[:-1])
        st.markdown(ans)
        st.session_state.messages.append({"role": "assistant", "content": ans})
