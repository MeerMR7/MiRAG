import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ---------------- UI ----------------
st.set_page_config(
    page_title="Sufiyan’s ChatBot | Academic Policy",
    page_icon="🤖"
)

st.title("🤖 Sufiyan’s ChatBot")
st.caption("Academic Policy Manual Assistant")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter OpenAI API Key", type="password")

# ---------------- RAG ENGINE ----------------
PDF_PATH = "Academic-Policy-Manual-for-Students2.pdf"

@st.cache_resource
def create_knowledge_base(pdf_path, key):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever()

# ---------------- CHAT MEMORY ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ---------------- CHAT INPUT ----------------
if prompt := st.chat_input("Ask about academic policies"):
    if not api_key:
        st.error("Please enter your OpenAI API key.")
    elif not os.path.exists(PDF_PATH):
        st.error("Academic-Policy-Manual-for-Students2.pdf not found.")
    else:
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Reading Academic Policy Manual..."):
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    openai_api_key=api_key
                )

                retriever = create_knowledge_base(PDF_PATH, api_key)

                system_prompt = (
                    "You are Sufiyan’s ChatBot, an academic policy assistant. "
                    "Answer ONLY using the Academic Policy Manual provided. "
                    "If the answer is not found, say you don't know.\n\n{context}"
                )

                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}")
                ])

                combine_chain = create_stuff_documents_chain(
                    llm, prompt_template
                )

                rag_chain = create_retrieval_chain(
                    retriever, combine_chain
                )

                response = rag_chain.invoke({"input": prompt})
                answer = response["answer"]

                st.write(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
