import fitz
from langchain_core.documents import Document


def load_pdf(file_path: str) -> list[Document]:
    """Extract PDF text and preserve page numbers."""
    documents = []

    with fitz.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_path,
                            "page": page_number,
                        },
                    )
                )

    if not documents:
        raise ValueError(
            "No readable text found. This PDF may be scanned "
            "and require OCR."
        )

    return documents