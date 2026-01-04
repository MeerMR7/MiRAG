import streamlit as st
import pdfplumber
import os
import re
from groq import Groq

# --- PASTE YOUR KEY HERE ---
MY_GROQ_KEY = "gsk_your_actual_key_here" 

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="HasMir's ChatBot", page_icon="🔮")

# 2. THEME CSS
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #111111; }
    .developer-tag { text-align: right; color: #888888; font-size: 14px; margin-top: -20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔮 HasMir's ChatBot")
st.markdown("<div class='developer-tag'>Developed By HasMir</div>", unsafe_allow_html=True)

# 3. SIDEBAR
with st.sidebar:
    st.header("Document Status")
    manual_path = "Academic-Policy-Manual-for-Students2.pdf"
    if os.path.exists(manual_path):
        st.success("✅ PDF Loaded")
    else:
        st.error("❌ PDF Missing")

# 4. RAG ENGINE
@st.cache_data
def get_chunks_from_pdf(path):
    chunks = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    chunks.extend(text.split('\n\n'))
        return chunks
    except: return []

def retrieve_relevant_chunks(query, chunks):
    query_words = set(re.findall(r'\w+', query.lower()))
    scored = []
    for c in chunks:
        score = len(query_words.intersection(set(re.findall(r'\w+', c.lower()))))
        if score > 0: scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scored[:5]]

# 5. INITIALIZATION
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_chunks" not in st.session_state and os.path.exists(manual_path):
    st.session_state.pdf_chunks = get_chunks_from_pdf(manual_path)

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 6. CHAT INTERACTION
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        try:
            # USES THE HARDCODED KEY DIRECTLY
            client = Groq(api_key=MY_GROQ_KEY)
            
            context = "\n\n".join(retrieve_relevant_chunks(prompt, st.session_state.get("pdf_chunks", [])))
            
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"You are HasMir's ChatBot. Use this context: {context}"},
                    *st.session_state.messages
                ],
                model="llama-3.1-70b-versatile",
            )
            
            answer = response.choices[0].message.content
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Error: {e}")
