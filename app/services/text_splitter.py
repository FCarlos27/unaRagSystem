from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils.logging import get_logger

logger = get_logger("text_splitter")


def split_documents(documents: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    """Chunk documents; passes prechunked entries through and splits the rest."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        # 1. Pass pre-chunked documents through as a single chunk
        if doc["metadata"].get("prechunked"):
            chunks.append(
                {
                    "page_content": doc["page_content"],
                    "metadata": {**doc["metadata"], "chunk_index": 0},
                }
            )
            continue

        # 2. Split standard document text
        pieces = splitter.split_text(doc["page_content"])
        for idx, piece in enumerate(pieces):
            chunks.append(
                {
                    "page_content": piece,
                    "metadata": {**doc["metadata"], "chunk_index": idx},
                }
            )

    logger.info("Split %d documents into %d chunks", len(documents), len(chunks))
    return chunks
