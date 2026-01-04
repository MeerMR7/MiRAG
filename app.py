import streamlit as st
import pdfplumber
import os
import re
from groq import Groq

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="HasMir's ChatBot", page_icon="🔮")

# 2. PURE BLACK THEME CSS
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
    
    /* Force white text for all elements */
    h1, h2, h3, p, span, label, .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Text input styling */
    .stTextInput>div>div>input {
        background-color: #222222 !important;
        color: white !important;
        border: 1px solid #444 !important;
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

st.title("🔮 HasMir's ChatBot")
st.markdown("<div class='developer-tag'>Developed By HasMir</div>", unsafe_allow_html=True)
st.write("### RAG Assistant: Academic Policy Expert")
st.markdown("---")

# 3. SIDEBAR
with st.sidebar:
    st.header("Setup")
    groq_key = st.text_input("Enter Groq API Key", type="password")
    
    st.subheader("Document Status")
    manual_path = "Academic-Policy-Manual-for-Students2.pdf"
    
    if os.path.exists(manual_path):
        st.success(f"✅ Document Active: {manual_path}")
    else:
        st.error(f"❌ Document Not Found: Please upload {manual_path} to your GitHub repository.")

# 4. RAG ENGINE (Retrieval Logic)
def get_chunks_from_pdf(path):
    chunks = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    # Chunks are created by paragraph (double newline)
                    paragraphs = text.split('\n\n')
                    chunks.extend(paragraphs)
        return chunks
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return []

def retrieve_relevant_chunks(query, chunks, top_n=5):
    # Simple Keyword-based Retrieval (TF-IDF light)
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

# Load chunks into memory once to keep the bot fast
if "pdf_chunks" not in st.session_state and os.path.exists(manual_path):
    st.session_state.pdf_chunks = get_chunks_from_pdf(manual_path)

# Display Chat History
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 6. CHAT INTERACTION
if prompt := st.chat_input("Ask a question about the policy..."):
    if not groq_key:
        st.error("Please provide your Groq API Key in the sidebar.")
    else:
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("HasMir's ChatBot is searching the manual..."):
                try:
                    # STEP 1: RETRIEVAL
                    context_chunks = retrieve_relevant_chunks(prompt, st.session_state.get("pdf_chunks", []))
                    context_text = "\n\n".join(context_chunks)
                    
                    # STEP 2: GENERATION (Groq)
                    client = Groq(api_key=groq_key)
                    system_instructions = (
                        "You are HasMir's ChatBot, a professional academic policy advisor. "
                        "Use the provided context from the Academic Policy Manual to answer. "
                        "If the answer is not in the context, clearly state that the manual "
                        "does not contain that information.\n\n"
                        f"CONTEXT FROM MANUAL:\n{context_text[:12000]}"
                    )

                    response = client.chat.completions.create(
                        messages=[{"role": "system", "content": system_instructions}, *st.session_state.messages],
                        model="llama-3.1-70b-versatile",
                    )
                    
                    answer = response.choices[0].message.content
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"Execution Error: {str(e)}")
