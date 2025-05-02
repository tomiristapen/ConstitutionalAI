import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os

# --- Interface ---
st.title("KZ AI Assistant on the Constitution of Kazakhstan")
query = st.text_input("Ask a question about the Constitution:")

# --- Use two models: one for LLM, another for embeddings ---
llm = OllamaLLM(model="llama3.2")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# --- Cache and load vector store ---
@st.cache_resource
def load_vectorstore():
    if not os.path.exists("constitution.txt"):
        st.error("The file 'constitution.txt' was not found!")
        return None

    loader = TextLoader("constitution.txt")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="chroma_db")
    vectorstore.persist()
    return vectorstore

vectorstore = load_vectorstore()

# --- Retrieval QA chain ---
if vectorstore:
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    if query:
        with st.spinner("Generating answer..."):
            result = qa.run(query)
        st.markdown("### 📌 Answer:")
        st.write(result)

# --- Upload additional .txt document ---
st.markdown("---")
st.subheader("📄 Upload your own text file (.txt)")
uploaded_file = st.file_uploader("Upload a .txt file", type="txt")

if uploaded_file:
    user_text = uploaded_file.read().decode("utf-8")
    st.text_area("File content:", user_text[:1000] + "..." if len(user_text) > 1000 else user_text)
