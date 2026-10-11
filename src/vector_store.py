
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


def create_vector_store(
    documents: list[Document],
    embeddings: OllamaEmbeddings,
) -> Chroma:
    """Store document chunks in a persistent ChromaDB database."""

    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="pdf_collection",
        persist_directory="data/chroma_db",
    )
