from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.table_extractor import extract_contact_table_entries
from app.utils.logging import get_logger

logger = get_logger("text_splitter")


def split_documents(documents: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    """Chunk documents; passes prechunked entries through and splits the rest."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = []
    for doc in documents:
        if doc["metadata"].get("prechunked"):
            chunks.append(
                {
                    "page_content": doc["page_content"],
                    "metadata": {**doc["metadata"], "chunk_index": len(chunks)},
                }
            )
            continue

        table_entries = extract_contact_table_entries(doc["page_content"], doc["metadata"]["source"])
        for entry in table_entries:
            chunks.append(
                {
                    "page_content": entry["page_content"],
                    "metadata": {**entry["metadata"], "chunk_index": len(chunks)},
                }
            )

        pieces = splitter.split_text(doc["page_content"])
        for piece in pieces:
            chunks.append(
                {
                    "page_content": piece,
                    "metadata": {**doc["metadata"], "chunk_index": len(chunks)},
                }
            )
    logger.info("Split %d documents into %d chunks", len(documents), len(chunks))
    return chunks
