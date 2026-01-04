import streamlit as st
import pdfplumber
import os
import re
from groq import Groq

# ==========================================
# 1. API & CONFIGURATION (HARDCODED)
# ==========================================
GROQ_API_KEY = "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"
PDF_FILENAME = "Academic-Policy-Manual-for-Students2.pdf"

st.set_page_config(page_title="HasMir's ChatBot", page_icon="🔮")

# ==========================================
# 2. PURE BLACK THEME UI
# ==========================================
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
    /* Chat message styling */
    .stChatMessage { background-color: #111111 !important; border-radius: 10px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔮 HasMir's ChatBot")
st.markdown("<div class='developer-tag'>Developed By HasMir</div>", unsafe_allow_html=True)
st.write("### RAG Assistant: Academic Policy Expert")
st.markdown("---")

# ==========================================
# 3. RAG ENGINE (PDF Retrieval Logic)
# ==========================================

# Function to extract text and break into chunks
def get_pdf_chunks(path):
    chunks = []
    if os.path.exists(path):
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        # Split by double newline to keep paragraphs together
                        paragraphs = text.split('\n\n')
                        chunks.extend(paragraphs)
            return chunks
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return []
    return []

# Function to find the most relevant chunks based on user query
def retrieve_context(query, chunks, top_n=5):
    query_words = set(re.findall(r'\w+', query.lower()))
    scored_chunks = []
    
    for chunk in chunks:
        chunk_words = set(re.findall(r'\w+', chunk.lower()))
        # Count overlapping words (Basic Keyword Search)
        score = len(query_words.intersection(chunk_words))
        if score > 0:
            scored_chunks.append((score, chunk))
    
    # Sort by highest score first
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [c[1] for c in scored_chunks[:top_n]]

# ==========================================
# 4. SESSION INITIALIZATION
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load the PDF content into memory only once
if "pdf_data" not in st.session_state:
    with st.spinner("Initializing Knowledge Base..."):
        st.session_state.pdf_data = get_pdf_chunks(PDF_FILENAME)

# ==========================================
# 5. SIDEBAR & STATUS
# ==========================================
with st.sidebar:
    st.header("HasMir Control Panel")
    if os.path.exists(PDF_FILENAME):
        st.success(f"✅ Knowledge Source: {PDF_FILENAME}")
    else:
        st.error(f"❌ Missing: {PDF_FILENAME}")
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# 6. CHAT INTERFACE & LOGIC
# ==========================================

# Display previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User Input
if prompt := st.chat_input("Ask a question about the Academic Policy..."):
    # 1. Store User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("HasMir's ChatBot is analyzing the manual..."):
            try:
                # 2. RETRIEVAL (Get relevant chunks from PDF)
                relevant_context = retrieve_context(prompt, st.session_state.pdf_data)
                context_str = "\n\n".join(relevant_context)
                
                # 3. GENERATION (Send context + question to Groq)
                client = Groq(api_key=GROQ_API_KEY)
                
                system_prompt = (
                    "You are HasMir's ChatBot, a professional academic policy advisor. "
                    "Instructions:\n"
                    "1. Use the provided context from the student manual to answer.\n"
                    "2. If the answer is not in the context, say: 'The manual does not provide information on this.'\n"
                    "3. Keep answers clear and formatted with bullet points if needed.\n\n"
                    f"CONTEXT FROM PDF:\n{context_str[:12000]}"
                )

                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *st.session_state.messages
                    ],
                    model="llama-3.1-70b-versatile",
                )
                
                # 4. Display & Store Assistant Response
                answer = response.choices[0].message.content
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"System Error: {str(e)}")
