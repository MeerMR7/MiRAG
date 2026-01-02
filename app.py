import os
import streamlit as st
import pdfplumber
import requests
import datetime

# -------------------- 1. BRANDING & CONFIG --------------------
st.set_page_config(page_title="MiRAG | Academic Assistant", layout="centered", page_icon="🎓")

# Securely fetch API Key
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Exact filename of your policy document
PDF_FILENAME = "Academic-Policy-Manual-for-Students2.pdf"
MODEL_NAME = "llama-3.1-8b-instant"

# -------------------- 2. PDF DATA RETRIEVAL LOGIC --------------------
@st.cache_data(show_spinner="MiRAG is indexing the Academic Policy Manual...")
def load_policy_manual():
    """Reads the PDF and breaks it into searchable chunks."""
    if not os.path.exists(PDF_FILENAME):
        return None
    
    all_text = ""
    try:
        with pdfplumber.open(Academic-Policy-Manual-for-Students2) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    all_text += page_text + "\n"
        
        # Split text into small chunks so the AI can find specific sections easily
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

# Initialize the policy data
policy_data = load_policy_manual()

def find_relevant_context(query):
    """Simple keyword-based search to find the right policy sections."""
    if not policy_data:
        return "No policy manual found. Answering from general knowledge."
    
    query_words = set(query.lower().split())
    # Score each chunk based on how many words from the question it contains
    scored_chunks = []
    for chunk in policy_data:
        score = len(query_words & set(chunk.lower().split()))
        scored_chunks.append((score, chunk))
    
    # Sort by highest score and take the top 3 most relevant chunks
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top_context = [c[1] for c in scored_chunks[:3] if c[0] > 0]
    return "\n\n".join(top_context)

# -------------------- 3. CHAT INTELLIGENCE --------------------
def mirag_chat(question, history):
    # Retrieve relevant data from the PDF for this specific question
    context = find_relevant_context(question)
    today = datetime.datetime.now().strftime("%d %B %Y")
    
    system_prompt = f"""
    You are MiRAG, an Academic Assistant created by Mir MUHAMMAD Rafique and Hasnain Ali Raza.
    Today's Date: {today}.
    
    Use the following sections from the Academic Policy Manual to answer the user's question accurately.
    If the answer is not in the manual, tell the user you don't have that specific data.
    
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
st.title("🤖 MiRAG: Policy Assistant")
st.subheader("Developed by Mir MUHAMMAD Rafique & Hasnain Ali Raza")

with st.sidebar:
    st.header("System Status")
    if policy_data:
        st.success(f"✅ {Academic-Policy-Manual-for-Students2} Loaded")
        st.info(f"Retrieved {len(policy_data)} policy sections.")
    else:
        st.error("❌ Manual Not Found")
        st.warning(f"Please ensure '{PDF_FILENAME}' is in the same folder as this script.")

# Chat History setup
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Assalam o Alaikum! I am MiRAG. I have indexed the Policy Manual. Ask me anything about university rules!"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if user_input := st.chat_input("Ask a policy question (e.g., 'What is the attendance rule?')"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching Manual..."):
            ans = mirag_chat(user_input, st.session_state.messages[:-1])
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})
