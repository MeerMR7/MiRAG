import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

# 1. UI CONFIGURATION
st.set_page_config(page_title="Sufiyan's ChatBot", page_icon="🤖")

# 2. CUSTOM CSS FOR BLACK THEME
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #000000 !important;
        color: #ffffff;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #111111 !important;
    }
    
    /* Force white text on headers and labels */
    h1, h2, h3, p, span, label, .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Input field styling */
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

# 3. SIDEBAR SETTINGS
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    
    st.subheader("Your Documents")
    uploaded_pdfs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# 4. HELPER FUNCTION (PDF Text Extraction)
def get_pdf_text(pdfs):
    text = ""
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# 5. CHAT LOGIC
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle User Input
if prompt := st.chat_input("Ask Sufiyan's ChatBot a question..."):
    if not api_key:
        st.error("Please provide an API Key in the sidebar.")
    else:
        # Initialize OpenAI Client
        client = OpenAI(api_key=api_key)
        
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Sufiyan's ChatBot is reading..."):
                try:
                    # Context building
                    context = ""
                    if uploaded_pdfs:
                        context = get_pdf_text(uploaded_pdfs)
                    
                    # Construct messages for OpenAI
                    messages_for_api = [
                        {"role": "system", "content": f"You are Sufiyan's ChatBot. Use the following context to answer if possible: {context[:15000]}"},
                    ]
                    # Add history
                    for m in st.session_state.messages:
                        messages_for_api.append(m)

                    # API Call
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages_for_api
                    )
                    
                    full_res = response.choices[0].message.content
                    st.write(full_res)
                    st.session_state.messages.append({"role": "assistant", "content": full_res})
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
