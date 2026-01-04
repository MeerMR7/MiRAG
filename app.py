import streamlit as st
import pdfplumber
import os
import re
from groq import Groq

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Mir MUHAMMAD Rafique & Co | Academic Advisor", 
    page_icon="📜",
    layout="centered"
)

# 2. PROFESSIONAL LIGHT THEME CSS
st.markdown("""
    <style>
    /* Main app background - Professional White/Grey */
    .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }

    /* Professional Footer Branding */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #ffffff;
        color: #1e293b;
        text-align: center;
        padding: 12px;
        font-weight: bold;
        font-size: 14px;
        border-top: 1px solid #e2e8f0;
        z-index: 100;
    }
    
    /* Clean up default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    <div class='footer'>Developed By Mir MUHAMMAD Rafique And Co</div>
    """, unsafe_allow_html=True)

# 3. RAG ENGINE (PDF Extraction & Retrieval)
@st.cache_resource
def load_and_chunk_pdf(file_path):
    """Extracts text from the PDF and breaks it into paragraphs."""
    chunks = []
    if not os.path.exists(file_path):
        return None
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    # Split into paragraphs to create manageable chunks
                    paragraphs = text.split('\n\n')
                    chunks.extend([p.strip() for p in paragraphs if p.strip()])
        return chunks
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return []

def get_relevant_context(query, chunks, top_n=5):
    """Simple keyword matching to find relevant paragraphs."""
    query_words = set(re.findall(r'\w+', query.lower()))
    scored_chunks = []
    for chunk in chunks:
        chunk_words = set(re.findall(r'\w+', chunk.lower()))
        score = len(query_words.intersection(chunk_words))
        if score > 0:
            scored_chunks.append((score, chunk))
    
    # Sort by score and take the best matches
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return "\n\n".join([c[1] for c in scored_chunks[:top_n]])

# 4. INITIALIZATION
MANUAL_FILE = "Academic-Policy-Manual-for-Students2.pdf"
st.title("📜 Academic Policy Advisor")
st.caption("Strategic Information Retrieval System | Mir MUHAMMAD Rafique & Co")

# Load data
if "pdf_chunks" not in st.session_state:
    with st.spinner("Processing Academic Manual..."):
        st.session_state.pdf_chunks = load_and_chunk_pdf(MANUAL_FILE)

if st.session_state.pdf_chunks is None:
    st.error(f"Critical Error: '{MANUAL_FILE}' not found in the repository.")
    st.stop()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. CHAT LOGIC
if prompt := st.chat_input("Ask about university policies..."):
    # Securely pull the key from Streamlit Cloud Secrets
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except KeyError:
        st.error("API Key Missing: Please add GROQ_API_KEY to 'Manage App Secrets'.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # 1. Retrieve matching text from PDF
            context = get_relevant_context(prompt, st.session_state.pdf_chunks)
            
            # 2. Build the AI prompt
            client = Groq(api_key=api_key)
            system_instructions = (
                "You are the Mir MUHAMMAD Rafique & Co AI Advisor. "
                "Provide professional, accurate answers based on the Academic Policy Manual. "
                "Use the provided context to answer. If the information is not there, "
                "politely direct the user to the Registrar's office.\n\n"
                f"POLICY CONTEXT:\n{context[:12000]}"
            )

            # 3. Stream response from Groq
            stream = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_instructions},
                    *st.session_state.messages
                ],
                model="llama-3.1-70b-versatile",
                stream=True,
            )
            
            full_response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"System Error: {str(e)}")
