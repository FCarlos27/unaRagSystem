from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.utils.logging import get_logger

logger = get_logger("text_splitter")


def _split_markdown(
    document: dict, chunk_size: int, chunk_overlap: int
) -> list[dict]:
    """Split markdown by headers so chunks are section-aware."""
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ],
        strip_headers=False,
    )
    refine = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for section in splitter.split_text(document["page_content"]):
        section_meta = section.metadata
        pieces = refine.split_text(section.page_content)
        for idx, piece in enumerate(pieces):
            chunks.append(
                {
                    "page_content": piece,
                    "metadata": {
                        **document["metadata"],
                        **section_meta,
                        "chunk_index": idx,
                    },
                }
            )
    return chunks


def split_documents(documents: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    """Chunk documents; markdown is header-split, everything else recursive."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        # 1. Split markdown documents by headers into section-aware chunks
        if doc["metadata"].get("format") == "markdown":
            chunks.extend(_split_markdown(doc, chunk_size, chunk_overlap))
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
