import os
import streamlit as st
import pdfplumber
import requests
import datetime

# -------------------- 1. BRANDING & CONFIG --------------------
st.set_page_config(page_title="MiRAG | Academic Assistant", layout="centered", page_icon="🤖")

# Securely fetch API Key
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "llama-3.1-8b-instant"

# -------------------- 2. SMART PDF LOADING (THE FIX) --------------------
@st.cache_data(show_spinner="MiRAG is indexing the Policy Manual...")
def process_pdf(file_obj):
    text = ""
    try:
        with pdfplumber.open(file_obj) as pdf:
            for page in pdf.pages:
                tx = page.extract_text()
                if tx: text += tx + "\n"
        
        # Chunking for better answers
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        chunks, buf = [], ""
        for line in lines:
            if len(buf) + len(line) <= 600:
                buf += " " + line
            else:
                chunks.append(buf.strip())
                buf = line
        if buf: chunks.append(buf.strip())
        return chunks
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return []

# -------------------- 3. USER INTERFACE (SIDEBAR) --------------------
with st.sidebar:
    st.title("⚙️ MiRAG Settings")
    st.markdown("---")
    
    # PDF PATH HANDLING: Either upload or auto-detect
    st.info("Step 1: Provide the Policy Manual")
    uploaded_file = st.file_uploader("Upload 'Academic-Policy-Manual.pdf'", type="pdf")
    
    st.markdown("---")
    st.markdown("### **Developers:**")
    st.write("👤 **Mir MUHAMMAD Rafique**")
    st.write("👤 **Hasnain Ali Raza**")
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# -------------------- 4. LOGIC & RETRIEVAL --------------------
policy_data = []
if uploaded_file:
    policy_data = process_pdf(uploaded_file)
else:
    st.warning("👈 Please upload the PDF in the sidebar to begin.")
    st.stop()

def get_context(query):
    q_words = set(query.lower().split())
    scored = []
    for c in policy_data:
        score = len(q_words & set(c.lower().split()))
        if score > 0: scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return "\n\n".join([c for score, c in scored[:3]])

# -------------------- 5. CHAT SYSTEM --------------------
def mirag_chat(question, history):
    context = get_context(question)
    today = datetime.datetime.now().strftime("%d %B %Y")
    
    system_prompt = f"""
    You are MiRAG (Mir MUHAMMAD Rafique and Hasnain Ali Raza's Chat Bot).
    Context from Manual: {context}
    
    Instructions: Provide direct answers based on the manual. 
    If not found, ask the user to clarify.
    """
    
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": question}]
    
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json={"model": MODEL_NAME, "messages": messages, "temperature": 0.3},
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"}
        )
        return r.json()["choices"][0]["message"]["content"]
    except:
        return "⚠️ Connection Error. Check your API Key."

# --- MAIN CHAT DISPLAY ---
st.title("🤖 MiRAG")
st.subheader("Mir MUHAMMAD Rafique & Hasnain Ali Raza's ChatBot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask about University Policy..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        ans = mirag_chat(user_input, st.session_state.messages[:-1])
        st.markdown(ans)
        st.session_state.messages.append({"role": "assistant", "content": ans})
