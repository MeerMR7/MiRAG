import streamlit as st
import pdfplumber
import os
import re
from groq import Groq

# 1. HARDCODED CONFIGURATION
# Your key is here. NOTE: GitHub will block this until you click "Allow"
GROQ_API_KEY = "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"
PDF_FILENAME = "Academic-Policy-Manual-for-Students2.pdf"

st.set_page_config(page_title="HasMir's ChatBot", page_icon="🔮")

# 2. UI THEME
st.markdown("""
    <style>
    .stApp { background-color: #000000 !important; color: #ffffff !important; }
    [data-testid="stSidebar"] { background-color: #000000 !important; border-right: 1px solid #333; }
    h1, h2, h3, p, span, label, .stMarkdown { color: #ffffff !important; }
    .stTextInput>div>div>input {
        background-color: #111111 !important;
        color: white !important;
        border: 1px solid #444 !important;
    }
    .developer-tag { text-align: right; color: #888888; font-size: 14px; margin-top: -20px; }
    hr { border-top: 1px solid #333 !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔮 HasMir's ChatBot")
st.markdown("<div class='developer-tag'>Developed By HasMir</div>", unsafe_allow_html=True)
st.write("### RAG Assistant: Academic Policy Expert")
st.markdown("---")

# 3. RAG ENGINE
def get_pdf_chunks(path):
    chunks = []
    if os.path.exists(path):
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        chunks.extend(text.split('\n\n'))
            return chunks
        except: return []
    return []

def retrieve_context(query, chunks, top_n=5):
    query_words = set(re.findall(r'\w+', query.lower()))
    scored_chunks = []
    for chunk in chunks:
        chunk_words = set(re.findall(r'\w+', chunk.lower()))
        score = len(query_words.intersection(chunk_words))
        if score > 0:
            scored_chunks.append((score, chunk))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [c[1] for c in scored_chunks[:top_n]]

# 4. INITIALIZATION
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = get_pdf_chunks(PDF_FILENAME)

# 5. SIDEBAR STATUS
with st.sidebar:
    st.header("System Status")
    if os.path.exists(PDF_FILENAME):
        st.success("✅ Manual Active")
    else:
        st.error(f"❌ {PDF_FILENAME} not found.")

# 6. CHAT LOGIC
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask a question about the policy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("HasMir's ChatBot is analyzing..."):
            try:
                # Get relevant context from PDF
                relevant_context = retrieve_context(prompt, st.session_state.pdf_data)
                context_str = "\n\n".join(relevant_context)
                
                # FIX FOR PROXIES ERROR: Initialize clean client
                client = Groq(api_key=GROQ_API_KEY)
                
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": f"You are HasMir's ChatBot. Use this context: {context_str[:10000]}"},
                        *st.session_state.messages
                    ],
                    model="llama-3.1-70b-versatile",
                )
                
                answer = response.choices[0].message.content
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
