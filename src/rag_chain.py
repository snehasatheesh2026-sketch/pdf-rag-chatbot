
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma


def answer_question(question: str, vector_store: Chroma) -> dict:
    """Answer questions using relevant passages from a PDF."""

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

    relevant_docs = retriever.invoke(question)

    if not relevant_docs:
        return {
            "answer": "I could not find the answer in the PDF.",
            "sources": [],
        }

    context_parts = []
    sources = []

    for doc in relevant_docs:
        page = doc.metadata.get("page", "Unknown")
        context_parts.append(
            f"[Page {page}]\n{doc.page_content}"
        )
        sources.append({
            "page": page,
            "text": doc.page_content,
        })

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an assistant that answers questions about a PDF.

Instructions:
- Read all the provided context carefully.
- Extract exact information, including names, email addresses,
  phone numbers, education, skills, and project details.
- For a person's name, identify the resume owner if the context
  provides that information.
- Answer directly and concisely.
- Use only the provided context. Never invent information.
- If the answer is missing, say you could not find it in the PDF.
- Cite the page number, such as [Page 1].

PDF CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    llm = ChatOllama(
        model="llama3.2",
        temperature=0,
    )

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": sources,
    }
