import os
import tempfile
import json
from datetime import datetime
import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from typing import List

DB_DIR = "chroma_db"
CONSTITUTION_FILE = "constitution.txt"
CHAT_HISTORY_DIR = "chat_logs"
SUPPORTED_FILE_TYPES = [".txt", ".pdf", ".docx"]

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "uploaded_texts" not in st.session_state:
        st.session_state.uploaded_texts = []
    if "context_mode" not in st.session_state:
        st.session_state.context_mode = "Only Constitution"
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = None
    if "selected_chat" not in st.session_state:
        st.session_state.selected_chat = None

def load_and_parse_constitution() -> List[Document]:
    loader = TextLoader(CONSTITUTION_FILE, encoding='utf-8')
    docs = loader.load()
    articles = []
    current_article = ""
    article_number = ""
    for line in docs[0].page_content.split('\n'):
        line = line.strip()
        if line.startswith("\u0421\u0442\u0430\u0442\u044c\u044f") or line.startswith("Article"):
            if current_article:
                articles.append(create_article_doc(current_article, article_number))
            parts = line.split()
            article_number = parts[1].replace(".", "") if len(parts) > 1 else "N/A"
            current_article = line + "\n"
        else:
            current_article += line + "\n"
    if current_article:
        articles.append(create_article_doc(current_article, article_number))
    return articles

def create_article_doc(content: str, article_num: str) -> Document:
    return Document(
        page_content=content.strip(),
        metadata={
            "source": "Constitution of Kazakhstan",
            "type": "constitution",
            "article": article_num
        }
    )

def process_uploaded_files(uploaded_files) -> List[Document]:
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    for uploaded_file in uploaded_files:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            if ext == ".txt":
                loader = TextLoader(tmp_path, encoding="utf-8")
            elif ext == ".pdf":
                loader = PyMuPDFLoader(tmp_path)
            elif ext == ".docx":
                loader = UnstructuredWordDocumentLoader(tmp_path)
            else:
                continue

            docs = loader.load()
            st.session_state.uploaded_texts.append(docs[0].page_content)
            file_chunks = splitter.split_documents(docs)
            for chunk in file_chunks:
                chunk.metadata.update({"source": uploaded_file.name, "type": "uploaded_file"})
            chunks.extend(file_chunks)
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
        finally:
            os.unlink(tmp_path)

    return chunks

def create_or_load_vector_store(docs: List[Document] = None):
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    if os.path.exists(DB_DIR) and not docs:
        return Chroma(persist_directory=DB_DIR, embedding_function=embedding)
    else:
        vs = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            persist_directory=DB_DIR
        )
        return vs

def get_prompt():
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a legal expert on Kazakhstan's Constitution. Use the context below to answer the question.

        {context}

        Question: {question}

        Response:
        """
    )

def list_chat_sessions():
    if not os.path.exists(CHAT_HISTORY_DIR):
        os.makedirs(CHAT_HISTORY_DIR)
    return sorted([f.replace(".json", "") for f in os.listdir(CHAT_HISTORY_DIR) if f.endswith(".json")])

def load_chat(chat_id):
    path = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_chat(chat_id, messages):
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
    path = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

def delete_chat(chat_id):
    path = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
    if os.path.exists(path):
        os.remove(path)

def reindex_chat_into_chroma(messages, vector_store):
    docs = []
    last_q = None
    for m in messages:
        if m["role"] == "user":
            last_q = m["content"]
        elif m["role"] == "assistant" and last_q:
            doc = Document(
                page_content=f"Q: {last_q}\nA: {m['content']}",
                metadata={"source": "chat", "type": "qa"}
            )
            docs.append(doc)
    if docs:
        vector_store.add_documents(docs)

def main():
    st.set_page_config(page_title="Kazakhstan Constitution QA", layout="wide")
    st.title("\U0001F1F0\U0001F1FF Ask About Kazakhstan's Constitution")
    initialize_session_state()

    with st.sidebar:
        st.subheader("\U0001F4C2 Upload Documents")
        uploaded_files = st.file_uploader("Drag and drop files here", type=SUPPORTED_FILE_TYPES, accept_multiple_files=True)
        if uploaded_files:
            uploaded_docs = process_uploaded_files(uploaded_files)
            if uploaded_docs:
                constitution_docs = load_and_parse_constitution()
                all_docs = uploaded_docs + constitution_docs
                st.session_state.vector_store = create_or_load_vector_store(all_docs)
                st.success("Files processed and indexed.")

        st.subheader("\U0001F4D1 Include uploaded files in analysis:")
        st.session_state.context_mode = st.radio("Choose context scope:", ["Only Constitution", "Combine with Uploaded Files"])

        st.subheader("\U0001F4AC Chat Sessions")
        chats = list_chat_sessions()
        st.session_state.selected_chat = st.selectbox("Open chat:", chats) if chats else None

        if st.session_state.selected_chat and st.session_state.selected_chat != st.session_state.chat_id:
            st.session_state.chat_id = st.session_state.selected_chat
            st.session_state.messages = load_chat(st.session_state.chat_id)
            if st.session_state.vector_store:
                reindex_chat_into_chroma(st.session_state.messages, st.session_state.vector_store)

        if st.button("\U0001F5D1️ Delete This Chat") and st.session_state.chat_id:
            delete_chat(st.session_state.chat_id)
            st.session_state.chat_id = None
            st.session_state.messages = []
            st.rerun()

        if st.button("\u2795 New Chat"):
            new_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            st.session_state.chat_id = new_id
            st.session_state.messages = []
            save_chat(new_id, [])
            st.rerun()

    if not st.session_state.vector_store:
        constitution_docs = load_and_parse_constitution()
        st.session_state.vector_store = create_or_load_vector_store(constitution_docs)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            llm = OllamaLLM(model="llama3.2", temperature=0.1)
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": get_prompt()}
            )

            if st.session_state.context_mode == "Combine with Uploaded Files" and st.session_state.uploaded_texts:
                uploaded_context = "\n---\n".join(st.session_state.uploaded_texts)
                enhanced_query = f"Uploaded content:\n{uploaded_context}\n\nQuestion:\n{user_input}"
            else:
                enhanced_query = user_input

            result = qa_chain({"query": enhanced_query})
            response = result["result"]
            sources = result.get("source_documents", [])

            if "not addressed" in response.lower():
                response = "\u2757 This is not addressed in the Constitution of Kazakhstan."

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response)

            if sources:
                with st.expander("\U0001F4D6 Source Documents"):
                    for doc in sources:
                        article = doc.metadata.get("article")
                        if article:
                            st.markdown(f"**Article {article}** — *{doc.metadata.get('source', '')}*\n\n{doc.page_content}")
                        else:
                            st.markdown(f"**{doc.metadata.get('source', '')}**\n\n{doc.page_content}")

            qa_doc = Document(
                page_content=f"Q: {user_input}\nA: {response}",
                metadata={"source": "chat", "type": "qa"}
            )
            st.session_state.vector_store.add_documents([qa_doc])

            if st.session_state.chat_id:
                save_chat(st.session_state.chat_id, st.session_state.messages)

if __name__ == "__main__":
    main()
