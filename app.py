
import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.embeddings import get_embeddings
from src.vector_store import create_vector_store
from src.rag_chain import answer_question


st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Chat with Your PDF")
st.caption("Ask questions about your PDF using a local Ollama model.")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None

if "messages" not in st.session_state:
    st.session_state.messages = []


uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type=["pdf"],
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    current_hash = hashlib.sha256(file_bytes).hexdigest()

    if current_hash != st.session_state.pdf_hash:
        with st.spinner("Reading PDF and building the search index..."):
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".pdf",
                    delete=False,
                ) as temp_file:
                    temp_file.write(file_bytes)
                    temp_path = temp_file.name

                try:
                    documents = load_pdf(temp_path)
                    chunks = split_documents(documents)

                    embeddings = get_embeddings()
                    vector_store = create_vector_store(
                        chunks,
                        embeddings,
                    )

                    st.session_state.vector_store = vector_store
                    st.session_state.pdf_hash = current_hash
                    st.session_state.messages = []

                finally:
                    Path(temp_path).unlink(missing_ok=True)

                st.success(
                    f"PDF processed successfully! "
                    f"{len(documents)} pages with text, "
                    f"{len(chunks)} text chunks."
                )

            except Exception as exc:
                st.error(f"Could not process the PDF: {exc}")
                st.stop()

if st.session_state.vector_store is not None:
    st.subheader("Ask your questions")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("View source passages"):
                    for source in message["sources"]:
                        st.markdown(f"**Page {source['page']}**")
                        st.write(source["text"])

    question = st.chat_input("Ask something about your PDF...")

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Finding relevant passages and generating an answer..."):
                try:
                    result = answer_question(
                        question,
                        st.session_state.vector_store,
                    )

                    st.markdown(result["answer"])

                    with st.expander("View source passages"):
                        for source in result["sources"]:
                            st.markdown(f"**Page {source['page']}**")
                            st.write(source["text"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    })

                except Exception as exc:
                    st.error(f"Could not generate an answer: {exc}")
else:
    st.info("Upload a PDF to get started.")
