import streamlit as st
import openai
from PyPDF2 import PdfReader

# ---------------- UI ----------------
st.set_page_config(
    page_title="Sufiyan’s ChatBot | PDF Chat",
    page_icon="🤖"
)

# Black background
st.markdown("""
<style>
body {
    background-color: #000000;
    color: white;
}
[data-testid="stAppViewContainer"] {
    background-color: #000000;
}
[data-testid="stHeader"] {
    background-color: #000000;
}
[data-testid="stSidebar"] {
    background-color: #111111;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Sufiyan’s ChatBot")
st.markdown("<div style='text-align:right; color:gray;'>Developed By Sufiyan</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter OpenAI API Key", type="password")

    st.subheader("Upload PDF")
    pdf_file = st.file_uploader("Upload a PDF", type="pdf")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- PDF READER ----------------
def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# ---------------- CHAT MEMORY ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ---------------- CHAT INPUT ----------------
if prompt := st.chat_input("Ask a question about the uploaded PDF"):
    if not api_key:
        st.error("Please enter your OpenAI API Key.")
    elif not pdf_file:
        st.error("Please upload a PDF file.")
    else:
        openai.api_key = api_key

        pdf_text = read_pdf(pdf_file)

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Sufiyan’s ChatBot is reading your PDF..."):
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are Sufiyan’s ChatBot, an academic assistant. "
                                    "Answer ONLY using the content of the PDF uploaded by the user. "
                                    "If the answer is not in the document, say you don't know.\n\n"
                                    f"{pdf_text}"
                                )
                            },
                            {"role": "user", "content": prompt}
                        ]
                    )

                    answer = response["choices"][0]["message"]["content"]
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(str(e))
