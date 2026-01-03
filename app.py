import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

#UI
st.set_page_config(page_title="MiRAG | PDF Chat", page_icon="🔮")

st.markdown("""
    <style>
    .developer-tag { text-align: right; color: #6c757d; font-size: 14px; margin-top: -20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔮 MiRAG")
st.markdown("<div class='developer-tag'>Developed By HasMir</div>", unsafe_allow_html=True)
st.markdown("---")

#SIDEBAR
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    
    st.subheader("Your Documents")
    uploaded_pdfs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

#RAG ENGINE
def create_knowledge_base(pdfs, key):
    all_docs = []
    for pdf in pdfs:
        # Temporary save to allow PyPDFLoader to read it
        with open("temp.pdf", "wb") as f:
            f.write(pdf.getbuffer())
        loader = PyPDFLoader("temp.pdf")
        all_docs.extend(loader.load())
        os.remove("temp.pdf") # Clean up

    # Split: Break PDF text into 1000-character chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    final_chunks = text_splitter.split_documents(all_docs)

    # Embed & Store: Convert text to math (vectors) and save in FAISS
    embeddings = OpenAIEmbeddings(openai_api_key=key)
    vectorstore = FAISS.from_documents(final_chunks, embeddings)
    return vectorstore.as_retriever()

#CHAT LOGIC
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User Input
if prompt := st.chat_input("Ask a question about your PDFs"):
    if not api_key:
        st.error("Please provide an API Key in the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("MiRAG is scanning documents..."):
                try:
                    # Initialize Brain
                    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)
                    
                    if uploaded_pdfs:
                        # RAG Pipeline
                        retriever = create_knowledge_base(uploaded_pdfs, api_key)
                        
                        system_prompt = (
                            "You are MiRAG, a precise document assistant. "
                            "Use the context below to answer. If the answer isn't in the context, "
                            "say you don't know based on the files provided.\n\n{context}"
                        )
                        prompt_template = ChatPromptTemplate.from_messages([
                            ("system", system_prompt),
                            ("human", "{input}"),
                        ])
                        
                        combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
                        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
                        
                        response = rag_chain.invoke({"input": prompt})
                        full_res = response["answer"]
                    else:
                        # Fallback to general AI if no PDF
                        full_res = llm.invoke(prompt).content

                    st.write(full_res)
                    st.session_state.messages.append({"role": "assistant", "content": full_res})
                except Exception as e:
                    st.error(f"Error: {str(e)}")
