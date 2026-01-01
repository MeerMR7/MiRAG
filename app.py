import os
import streamlit as st
import pdfplumber
import requests
import datetime

# -------------------- 1. BRANDING & CONFIG --------------------
st.set_page_config(page_title="MiRAG | Academic Assistant", layout="centered", page_icon="🤖")

if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# The exact name of your file
PDF_FILENAME = "Academic-Policy-Manual-for-Students2.pdf"
MODEL_NAME = "llama-3.1-8b-instant"

# -------------------- 2. SMART PDF LOADING --------------------
@st.cache_data(show_spinner="MiRAG is indexing the Policy Manual...")
def process_pdf(source):
    text = ""
    try:
        # Works with both a file path (string) or an uploaded file object
        with pdfplumber.open(source) as pdf:
            for page in pdf.pages:
                tx = page.extract_text()
                if tx: text += tx + "\n"
        
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        chunks, buf = [], ""
        for line in lines:
            if len(buf) + len(line) <= 600:
                buf += " " + line
            else:
                chunks.append(buf.strip()); buf = line
        if buf: chunks.append(buf.strip())
        return chunks
    except Exception as e:
        return None

# -------------------- 3. PATH LOGIC (THE ERROR FIX) --------------------
# Check if file exists in the same folder as this script
local_path = os.path.join(os.getcwd(), PDF_FILENAME)

if os.path.exists(local_path):
    # Option A: Load automatically if found
    policy_data = process_pdf(local_path)
    st.sidebar.success(f"✅ Found: {PDF_FILENAME}")
else:
    # Option B: Ask user to upload if not found (Prevents Error)
    st.sidebar.warning(f"⚠️ '{PDF_FILENAME}' not found in folder.")
    uploaded_file = st.sidebar.file_uploader("Please upload the PDF manually:", type="pdf")
    if uploaded_file:
        policy_data = process_pdf(uploaded_file)
    else:
        st.info("👈 Please upload the policy PDF in the sidebar to activate MiRAG.")
        st.stop()

# -------------------- 4. CHAT LOGIC --------------------
def get_context(query):
    q_words = set(query.lower().split())
    scored = sorted([(len(q_words & set(c.lower().split())), c) for c in policy_data], reverse=True)
    return "\n\n".join([c for score, c in scored[:3]])

def mirag_chat(question, history):
    context = get_context(question)
    today = datetime.datetime.now().strftime("%d %B %Y")
    system_prompt = f"You are MiRAG (By Mir MUHAMMAD Rafique & Hasnain Ali Raza). Context: {context}"
    
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": question}]
    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", 
                          json={"model": MODEL_NAME, "messages": messages, "temperature": 0.3},
                          headers={"Authorization": f"Bearer {GROQ_API_KEY}"})
        return r.json()["choices"][0]["message"]["content"]
    except:
        return "⚠️ Connection Error."

# -------------------- 5. USER INTERFACE --------------------
st.title("🤖 MiRAG")
st.subheader("Mir MUHAMMAD Rafique & Hasnain Ali Raza's ChatBot")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Assalam o Alaikum! How can I help you with the policy?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if user_input := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)
    with st.chat_message("assistant"):
        ans = mirag_chat(user_input, st.session_state.messages[:-1])
        st.markdown(ans)
        st.session_state.messages.append({"role": "assistant", "content": ans})
