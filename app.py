import os
import streamlit as st
import pdfplumber
import requests
import datetime

# -------------------- 1. BRANDING & CONFIG --------------------
st.set_page_config(page_title="MiRAG | Academic Assistant", layout="centered", page_icon="🎓")

# API Key Handling
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Exact filename of your policy document
PDF_PATH = "Academic-Policy-Manual-for-Students2.pdf"
MODEL_NAME = "llama-3.1-8b-instant"

# -------------------- 2. POLICY DATA RETRIEVAL --------------------
@st.cache_data(show_spinner="MiRAG is indexing the Policy Manual...")
def load_policy_manual():
    """Reads the PDF and creates searchable data chunks."""
    if not os.path.exists(PDF_FILENAME):
        return None
    
    all_text = ""
    try:
        with pdfplumber.open(PDF_FILENAME) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    all_text += page_text + "\n"
        
        # Break manual into sections for easier searching
        lines = [line.strip() for line in all_text.split("\n") if line.strip()]
        chunks, current_chunk = [], ""
        for line in lines:
            if len(current_chunk) + len(line) < 700:
                current_chunk += " " + line
            else:
                chunks.append(current_chunk.strip())
                current_chunk = line
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    except Exception as e:
        st.sidebar.error(f"Error reading PDF: {e}")
        return None

# Load the data
policy_data = load_policy_manual()

def get_relevant_policy(query):
    """Finds the most relevant sections of the manual for a user question."""
    if not policy_data:
        return ""
    
    query_words = set(query.lower().split())
    scored_chunks = []
    for chunk in policy_data:
        score = len(query_words & set(chunk.lower().split()))
        scored_chunks.append((score, chunk))
    
    # Sort and return the best matches
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top_context = [c[1] for c in scored_chunks[:3] if c[0] > 0]
    return "\n\n".join(top_context)

# -------------------- 3. CHAT INTELLIGENCE --------------------
def mirag_chat(question, history):
    # Retrieve data from the PDF based on the question
    context = get_relevant_policy(question)
    today = datetime.datetime.now().strftime("%d %B %Y")
    
    system_prompt = f"""
    You are MiRAG, an Academic Assistant for Iqra University students. 
    Created by: Mir MUHAMMAD Rafique and Hasnain Ali Raza.
    Date: {today}.
    
    INSTRUCTIONS: Use the POLICY CONTEXT below to answer the student's question. 
    If the context doesn't have the answer, politely explain that you only have 
    access to the student policy manual.

    POLICY CONTEXT:
    {context}
    """

    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": question}]

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions", 
            json={"model": MODEL_NAME, "messages": messages, "temperature": 0.2},
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"}
        )
        return response.json()["choices"][0]["message"]["content"]
    except:
        return "⚠️ Connection Error. Please verify your Groq API Key."

# -------------------- 4. USER INTERFACE --------------------
st.title("🤖 MiRAG: Academic Assistant")
st.subheader("Developed by Mir MUHAMMAD Rafique & Hasnain Ali Raza")

with st.sidebar:
    st.header("System Status")
    if policy_data:
        st.success(f"✅ {PDF_FILENAME} Active")
    else:
        st.error("❌ Policy Manual Not Found")
        st.info(f"Please ensure '{PDF_FILENAME}' is in your GitHub folder.")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Assalam o Alaikum! I am MiRAG. I have loaded the Academic Policy Manual. How can I help you today?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if user_input := st.chat_input("Ask a policy question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing manual..."):
            ans = mirag_chat(user_input, st.session_state.messages[:-1])
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})
