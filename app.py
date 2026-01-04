import streamlit as st
import pdfplumber
import os
from groq import Groq
from tavily import TavilyClient

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Sufiyan's ChatBot", page_icon="🎓")

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

st.title("🎓 Sufiyan's ChatBot")
st.markdown("<div class='developer-tag'>Developed By Sufiyan</div>", unsafe_allow_html=True)
st.write("Specializing in: *Academic Policy Manual for Students*")
st.markdown("---")

# 3. SIDEBAR
with st.sidebar:
    st.header("Setup")
    groq_key = st.text_input("Enter Groq API Key", type="password")
    tavily_key = st.text_input("Enter Tavily API Key (Optional)", type="password")
    
    st.subheader("Document Status")
    manual_path = "Academic-Policy-Manual-for-Students2.pdf"
    
    if os.path.exists(manual_path):
        st.success(f"✅ Found: {manual_path}")
    else:
        st.error(f"❌ Missing: {manual_path}. Please ensure it's in your GitHub root folder.")

# 4. HELPER FUNCTION: PDF EXTRACTION
@st.cache_data
def load_manual_content(path):
    full_text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        return full_text
    except Exception as e:
        return f"Error reading PDF: {e}"

# 5. CHAT INITIALIZATION
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 6. CHAT LOGIC
if prompt := st.chat_input("Ask about the Academic Policy..."):
    if not groq_key:
        st.error("Please enter your Groq API Key in the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Sufiyan's ChatBot is analyzing the manual..."):
                try:
                    # 1. Load context from the specific file
                    manual_content = load_manual_content(manual_path)
                    
                    # 2. Initialize Groq
                    client = Groq(api_key=groq_key)
                    
                    # 3. Build the prompt (Limiting context to stay within token limits)
                    system_prompt = (
                        "You are Sufiyan's ChatBot, an expert on student academic policies. "
                        "Use the following manual content to answer the user accurately. "
                        "If the answer is not in the manual, state that you don't know based on the document.\n\n"
                        f"MANUAL CONTEXT:\n{manual_content[:12000]}"
                    )

                    # 4. Call Groq
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            *st.session_state.messages
                        ],
                        model="llama-3.1-70b-versatile",
                    )
                    
                    response = chat_completion.choices[0].message.content
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
