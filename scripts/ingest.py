"""Ingestion script: load PDFs, chunk, embed, and store in ChromaDB."""

from app.core.config import get_settings
from app.services.document_loader import load_documents
from app.services.embeddings import build_embeddings
from app.services.text_splitter import split_documents
from app.services.vector_store import build_vector_store
from app.utils.logging import setup_logging

logger = setup_logging()


def main() -> None:
    settings = get_settings()

    documents = load_documents(settings.raw_pdfs_dir, settings.json_docs_dir)
    if not documents:
        logger.warning("No PDFs found in %s", settings.raw_pdfs_dir)
        return

    chunks = split_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    embeddings = build_embeddings(settings)
    vector_store = build_vector_store(settings, embeddings)

    vector_store.add_texts(
        texts=[chunk["page_content"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )
    logger.info("Indexed %d chunks into %s", len(chunks), settings.chroma_collection_name)


if __name__ == "__main__":
    main()
