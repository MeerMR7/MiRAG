import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from dotenv import load_dotenv

# === 1. MiRAG Configuration ===
st.set_page_config(page_title="MiRAG | Personal Assistant", page_icon="🤖", layout="wide")
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.markdown("<h2 style='text-align: center;'>🤖 MiRAG: My Intelligent RAG</h2>", unsafe_allow_html=True)
st.divider()

# === 2. Document Processing (Low Storage Optimized) ===
@st.cache_resource(show_spinner="🧠 Initializing MiRAG Brain...")
def prepare_vectorstore(uploaded_file):
    # Save temp file to read it
    with open("temp_doc.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    loader = PyPDFLoader("temp_doc.pdf")
    pages = loader.load()
    # Smaller chunks help with precision
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(pages)
    
    # Smallest high-quality embedding model available (~80MB)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

# Sidebar for Dynamic Uploads
with st.sidebar:
    st.header("📁 Personal Data")
    uploaded_file = st.file_uploader("Upload a PDF to MiRAG", type="pdf")
    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()

# === 3. Chat Logic ===
if uploaded_file:
    vectorstore = prepare_vectorstore(uploaded_file)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("Ask MiRAG something..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Retrieval & Generation
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are MiRAG, a precise personal assistant. Answer based ONLY on the provided context."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
            ],
            temperature=0.2
        )
        
        answer = response.choices[0].message.content
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👋 Welcome to MiRAG! Please upload a PDF in the sidebar to begin.")
