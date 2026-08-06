from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils.logging import get_logger

logger = get_logger("text_splitter")


def split_documents(documents: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = []
    for doc in documents:
        pieces = splitter.split_text(doc["page_content"])
        for i, piece in enumerate(pieces):
            chunks.append(
                {
                    "page_content": piece,
                    "metadata": {**doc["metadata"], "chunk_index": i},
                }
            )
    logger.info("Split %d documents into %d chunks", len(documents), len(chunks))
    return chunks
