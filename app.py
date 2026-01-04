import streamlit as st
import pdfplumber
import os
import re
from groq import Groq

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="HasMir's ChatBot", page_icon="🔮", layout="centered")

# 2. ENHANCED DARK THEME CSS
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid #333; }
    .developer-tag { text-align: right; color: #888888; font-size: 14px; margin-top: -20px; }
    
    /* Improve chat message styling */
    .stChatMessage { background-color: #111111; border-radius: 10px; margin-bottom: 10px; border: 1px solid #222; }
    footer {visibility: hidden;}
    hr { border-top: 1px solid #333 !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔮 HasMir's ChatBot")
st.markdown("<div class='developer-tag'>Developed By HasMir</div>", unsafe_allow_html=True)
st.write("### RAG Assistant: Academic Policy Expert")
st.markdown("---")

# 3. SIDEBAR SETUP
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_key = st.text_input("Groq API Key", type="password", help="Get your key from console.groq.com")
    
    st.subheader("📄 Document Status")
    manual_path = "Academic-Policy-Manual-for-Students2.pdf"
    
    if os.path.exists(manual_path):
        st.success(f"Active: {manual_path}")
    else:
        st.error(f"Missing: {manual_path}")
        st.info("Please ensure the PDF is in the same directory as this script.")

# 4. ENHANCED RAG ENGINE
@st.cache_data # Cache text extraction to save resources
def get_chunks_from_pdf(path):
    chunks = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    # Clean up whitespace and split by paragraphs or fixed length
                    text = re.sub(r'\s+', ' ', text)
                    # Create overlapping chunks (approx 500 chars) for better context
                    page_chunks = [text[i:i+600] for i in range(0, len(text), 400)]
                    chunks.extend(page_chunks)
        return chunks
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return []

def retrieve_relevant_chunks(query, chunks, top_n=5):
    if not chunks: return []
    
    # Improved Keyword Matching (Case-insensitive & Tokenized)
    query_words = set(re.findall(r'\w+', query.lower()))
    scored_chunks = []
    
    for chunk in chunks:
        chunk_words = set(re.findall(r'\w+', chunk.lower()))
        # Calculate intersection score
        score = len(query_words.intersection(chunk_words))
        if score > 0:
            scored_chunks.append((score, chunk))
    
    # Sort by highest score and take top N
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [c[1] for c in scored_chunks[:top_n]]

# 5. INITIALIZATION
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_chunks" not in st.session_state and os.path.exists(manual_path):
    with st.spinner("Indexing policy manual..."):
        st.session_state.pdf_chunks = get_chunks_from_pdf(manual_path)

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. CHAT INTERACTION
if prompt := st.chat_input("Ask about grading, attendance, or graduation..."):
    if not groq_key:
        st.warning("Please enter your Groq API Key in the sidebar.")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            with st.spinner("Analyzing manual..."):
                try:
                    # RETRIEVAL
                    context_chunks = retrieve_relevant_chunks(prompt, st.session_state.get("pdf_chunks", []))
                    context_text = "\n---\n".join(context_chunks)
                    
                    # GENERATION
                    client = Groq(api_key=groq_key)
                    
                    system_prompt = f"""You are HasMir's ChatBot, a helpful Academic Policy Advisor.
                    Rules:
                    1. Use ONLY the context below to answer. 
                    2. If the context doesn't have the answer, say: 'I'm sorry, the Academic Policy Manual does not contain information regarding that.'
                    3. Keep answers concise and use bullet points for lists.
                    
                    CONTEXT:
                    {context_text}"""

                    # Limit history to last 5 messages to save tokens/context
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            *st.session_state.messages[-5:] 
                        ],
                        model="llama-3.1-70b-versatile",
                        temperature=0.2, # Lower temperature for factual accuracy
                    )
                    
                    answer = chat_completion.choices[0].message.content
                    response_placeholder.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"Connection Error: {str(e)}")
