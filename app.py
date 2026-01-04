import streamlit as st
import requests
import io
import re
from groq import Groq
from pypdf import PdfReader

# 1. PAGE CONFIGURATION & BRANDING
st.set_page_config(
    page_title="Mir MUHAMMAD Rafique & Co | Advisor", 
    page_icon="📋",
    layout="centered"
)

# 2. PROFESSIONAL LIGHT THEME CSS (No Black Background)
st.markdown("""
    <style>
    .stApp {
        background-color: #f8fafc !important;
        color: #1e293b !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    
    .developer-tag { 
        text-align: center; 
        color: #64748b; 
        font-size: 14px; 
        font-weight: 600;
        padding: 20px;
    }

    /* Fixed Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #ffffff;
        color: #0f172a;
        text-align: center;
        padding: 10px;
        font-weight: bold;
        border-top: 1px solid #e2e8f0;
        z-index: 100;
    }
    </style>
    <div class='footer'>Developed By Mir MUHAMMAD Rafique And Co</div>
    """, unsafe_allow_html=True)

st.title("📋 Academic Policy Expert")
st.write("Retrieving policy data directly from GitHub repository...")
st.markdown("---")

# 3. GITHUB DATA RETRIEVAL
# Replace this with your actual RAW GitHub URL
GITHUB_PDF_URL = "https://raw.githubusercontent.com/YOUR_USER/YOUR_REPO/main/Academic-Policy-Manual-for-Students2.pdf"

@st.cache_data
def load_data_from_github(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            pdf_file = io.BytesIO(response.content)
            reader = PdfReader(pdf_file)
            chunks = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    # Split by double newline for paragraph chunks
                    paragraphs = text.split('\n\n')
                    chunks.extend(paragraphs)
            return chunks
        else:
            st.error("Could not find the PDF on GitHub. Check your URL.")
            return []
    except Exception as e:
        st.error(f"Error connecting to GitHub: {e}")
        return []

# 4. RAG ENGINE (Keyword Retrieval)
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

# Load chunks into memory
if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = load_data_from_github(GITHUB_PDF_URL)

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. CHAT INTERACTION
if prompt := st.chat_input("Ask a question about the policy..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing manual..."):
            try:
                # Retrieve Context
                context_chunks = retrieve_relevant_chunks(prompt, st.session_state.pdf_chunks)
                context_text = "\n\n".join(context_chunks)
                
                # Groq API Call
                client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                system_instructions = (
                    "You are a professional academic policy advisor for Mir MUHAMMAD Rafique And Co. "
                    "Use the following context from the manual to answer the user accurately. "
                    "If the answer isn't in the manual, say so politely.\n\n"
                    f"CONTEXT:\n{context_text[:10000]}"
                )

                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_instructions},
                        *st.session_state.messages
                    ],
                    model="llama-3.1-70b-versatile",
                    stream=True
                )
                
                # Stream the response for a professional feel
                full_response = st.write_stream(response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"System Error: {str(e)}")
