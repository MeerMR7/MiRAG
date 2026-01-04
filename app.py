import streamlit as st
import pdfplumber
import os
import re
from groq import Groq

# ==========================================
# 1. HARDCODED CONFIGURATION
# ==========================================
GROQ_API_KEY = "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"
PDF_FILENAME = "Academic-Policy-Manual-for-Students2.pdf"
# UPDATED MODEL NAME (llama-3.1-70b is decommissioned)
MODEL_NAME = "llama-3.3-70b-versatile" 

st.set_page_config(page_title="HasMir's ChatBot", page_icon="🔮")

# ==========================================
# 2. UI THEME
# ==========================================
st.markdown("""
    <style>
    .stApp { background-color: #000000 !important; color: #ffffff !important; }
    [data-testid="stSidebar"] { background-color: #000000 !important; border-right: 1px solid #333; }
    h1, h2, h3, p, span, label, .stMarkdown { color: #ffffff !important; }
    .stTextInput>div>div>input { background-color: #111111 !important; color: white !important; }
    .developer-tag { text-align: right; color: #888888; font-size: 14px; margin-top: -20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔮 HasMir's ChatBot")
st.markdown("<div class='developer-tag'>Developed By HasMir</div>", unsafe_allow_html=True)

# ==========================================
# 3. RAG ENGINE
# ==========================================
def load_pdf_data(path):
    if not os.path.exists(path): return []
    chunks = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text: chunks.extend(text.split('\n\n'))
        return chunks
    except: return []

# Initialize session
if "messages" not in st.session_state: st.session_state.messages = []
if "pdf_data" not in st.session_state: st.session_state.pdf_data = load_pdf_data(PDF_FILENAME)

# ==========================================
# 4. CHAT INTERFACE
# ==========================================
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask a policy question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        try:
            # 1. RETRIEVAL (Find matching paragraphs)
            query_words = set(re.findall(r'\w+', prompt.lower()))
            relevant = [c for c in st.session_state.pdf_data if query_words.intersection(set(re.findall(r'\w+', c.lower())))]
            context = "\n\n".join(relevant[:5])
            
            # 2. GENERATION (Call Groq)
            client = Groq(api_key=GROQ_API_KEY) # No proxies argument here
            
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"You are HasMir's ChatBot. Use this manual context: {context[:8000]}"},
                    *st.session_state.messages
                ],
                model=MODEL_NAME, # Using the new llama-3.3 model
            )
            
            answer = response.choices[0].message.content
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            st.error(f"System Error: {str(e)}")
