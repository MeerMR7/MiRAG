import streamlit as st
import pdfplumber
import os
import re
from groq import Groq

# --- SECURE API KEY SETUP ---
# This line tells the app to look for the key in your Streamlit Settings
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    st.error("Missing GROQ_API_KEY! Please add it to your Streamlit Cloud Secrets.")
    st.stop()

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="HasMir's ChatBot", page_icon="🔮")

# 2. PURE BLACK THEME CSS
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

# 3. SIDEBAR
with st.sidebar:
    st.header("System Status")
    manual_path = "Academic-Policy-Manual-for-Students2.pdf"
    
    if os.path.exists(manual_path):
        st.success("✅ Manual Active")
    else:
        st.error(f"❌ {manual_path} not found in GitHub.")
    
    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()

# 4. RAG ENGINE (Retrieval Logic)
def get_chunks_from_pdf(path):
    chunks = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    chunks.extend(text.split('\n\n'))
        return chunks
    except Exception:
        return []

def retrieve_relevant_chunks(query, chunks, top_n=5):
    query_words = set(re.findall(r'\w+', query.lower()))
    scored_chunks = []
    for chunk in chunks:
        chunk_words = set(re.findall(r'\w+', chunk.lower()))
        score = len(query_words.intersection(chunk_words))
        if score > 0:
            scored_chunks.append((score, chunk))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [c[1] for c in scored_chunks[:top_n]]

# 5. INITIALIZATION
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_chunks" not in st.session_state and os.path.exists(manual_path):
    st.session_state.pdf_chunks = get_chunks_from_pdf(manual_path)

# 6. CHAT INTERFACE
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask HasMir's ChatBot a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing manual..."):
            try:
                # RETRIEVAL
                relevant_info = retrieve_relevant_chunks(prompt, st.session_state.get("pdf_chunks", []))
                context = "\n\n".join(relevant_info)
                
                # GENERATION
                client = Groq(api_key=GROQ_API_KEY)
                system_instructions = (
                    "You are HasMir's ChatBot, an academic policy advisor. "
                    "Use the provided context to answer. If the answer isn't in the context, "
                    "say it isn't in the manual.\n\n"
                    f"CONTEXT FROM MANUAL:\n{context[:12000]}"
                )

                response = client.chat.completions.create(
                    messages=[{"role": "system", "content": system_instructions}, *st.session_state.messages],
                    model="llama-3.1-70b-versatile",
                )
                
                answer = response.choices[0].message.content
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
