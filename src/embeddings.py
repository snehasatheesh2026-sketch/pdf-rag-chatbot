
from langchain_ollama import OllamaEmbeddings


def get_embeddings() -> OllamaEmbeddings:
    """Create embeddings using the local Ollama model."""
    return OllamaEmbeddings(
        model="nomic-embed-text"
    )
