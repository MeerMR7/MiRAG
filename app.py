import streamlit as st
import pdfplumber
import os
import re
from groq import Groq

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="MiRAG | Academic Advisor", page_icon="🤖")

# 2. PURE BLACK THEME CSS (Polished)
st.markdown("""
    <style>
    /* Main app background */
    .stApp {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #111111 !important;
    }
    
    /* White text for visibility on black */
    h1, h2, h3, p, span, label, .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Chat message area styling */
    .stChatMessage {
        background-color: #111111 !important;
        border-radius: 10px;
        margin-bottom: 10px;
    }

    .developer-tag { 
        text-align: right; 
        color: #888888; 
        font-size: 14px; 
        margin-top: -20px; 
    }
    
    hr { border-top: 1px solid #333 !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 MiRAG")
st.markdown("<div class='developer-tag'>Developed By Mir MUHAMMAD Rafique And Co</div>", unsafe_allow_html=True)
st.write("### Academic Policy Expert")
st.markdown("---")

# 3. PDF CONFIGURATION
MANUAL_PATH = "Academic-Policy-Manual-for-Students2.pdf"

# 4. RAG ENGINE (Optimized)
@st.cache_resource
def get_chunks_from_pdf(path):
    if not os.path.exists(path):
        return None
    chunks = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    # Chunks are created by paragraph (double newline)
                    paragraphs = text.split('\n\n')
                    chunks.extend([p.strip() for p in paragraphs if p.strip()])
        return chunks
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return []

def retrieve_relevant_chunks(query, chunks, top_n=5):
    if not chunks:
        return ""
    # Simple Keyword-based Retrieval
    query_words = set(re.findall(r'\w+', query.lower()))
    scored_chunks = []
    for chunk in chunks:
        chunk_words = set(re.findall(r'\w+', chunk.lower()))
        score = len(query_words.intersection(chunk_words))
        if score > 0:
            scored_chunks.append((score, chunk))
    
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return "\n\n".join([c[1] for c in scored_chunks[:top_n]])

# 5. INITIALIZATION
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load chunks into memory once
if "pdf_chunks" not in st.session_state:
    with st.spinner("Loading Knowledge Base..."):
        st.session_state.pdf_chunks = get_chunks_from_pdf(MANUAL_PATH)

# Check if file exists
if st.session_state.pdf_chunks is None:
    st.error(f"❌ File Not Found: Please ensure '{MANUAL_PATH}' is in your GitHub repository.")
    st.stop()

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. CHAT INTERACTION
if prompt := st.chat_input("Ask a question about the policy..."):
    # Securely get API Key from Secrets
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except KeyError:
        st.error("API Key Missing: Please add GROQ_API_KEY to your Streamlit App Secrets.")
        st.stop()

    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant message
    with st.chat_message("assistant"):
        try:
            # STEP 1: RETRIEVAL
            context_text = retrieve_relevant_chunks(prompt, st.session_state.pdf_chunks)
            
            # STEP 2: GENERATION (Groq)
            client = Groq(api_key=api_key)
            system_instructions = (
                "You are MiRAG, the official AI Advisor for Mir MUHAMMAD Rafique And Co. "
                "Use the provided context from the Academic Policy Manual to answer. "
                "If the information is not in the context, inform the user politely and "
                "suggest they contact the Registrar's office.\n\n"
                f"CONTEXT FROM MANUAL:\n{context_text[:12000]}"
            )

            # Creating a stream for better UI
            stream = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_instructions},
                    *st.session_state.messages
                ],
                model="llama-3.1-70b-versatile",
                stream=True,
            )
            
            # Streaming the response to the UI
            full_response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Connection Error: {str(e)}")
