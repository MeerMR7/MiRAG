import streamlit as st
import pdfplumber
from groq import Groq
from tavily import TavilyClient

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Sufiyan's ChatBot", page_icon="🤖")

# 2. PURE BLACK THEME CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] {
        background-color: #111111 !important;
    }
    h1, h2, h3, p, span, label, .stMarkdown {
        color: #ffffff !important;
    }
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

st.title("🤖 Sufiyan's ChatBot")
st.markdown("<div class='developer-tag'>Developed By Sufiyan</div>", unsafe_allow_html=True)
st.markdown("---")

# 3. SIDEBAR
with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    tavily_api_key = st.text_input("Enter Tavily API Key (Optional)", type="password")
    
    st.subheader("Your Documents")
    uploaded_pdfs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# 4. HELPER FUNCTION: PDF EXTRACTION (pdfplumber)
def extract_text_from_pdf(pdfs):
    full_text = ""
    for pdf in pdfs:
        with pdfplumber.open(pdf) as pdf_doc:
            for page in pdf_doc.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
    return full_text

# 5. CHAT INITIALIZATION
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 6. USER INPUT & AI LOGIC
if prompt := st.chat_input("Ask Sufiyan's ChatBot..."):
    if not groq_api_key:
        st.error("Please provide a Groq API Key in the sidebar.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Sufiyan's ChatBot is analyzing..."):
                try:
                    # Initialize Groq Client
                    client = Groq(api_key=groq_api_key)
                    
                    # Get PDF Context
                    context = ""
                    if uploaded_pdfs:
                        context = extract_text_from_pdf(uploaded_pdfs)
                    
                    # Build System Prompt
                    system_message = f"You are Sufiyan's ChatBot. "
                    if context:
                        system_message += f"Use this PDF content to answer: {context[:15000]}"
                    else:
                        system_message += "Answer the user's questions clearly."

                    # API Call
                    response = client.chat.completions.create(
                        model="llama-3.1-70b-versatile",
                        messages=[
                            {"role": "system", "content": system_message},
                            *st.session_state.messages
                        ]
                    )
                    
                    answer = response.choices[0].message.content
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
