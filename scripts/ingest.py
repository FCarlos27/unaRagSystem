"""Ingestion script: load DOCX files, chunk, embed, and store in ChromaDB."""

import argparse

from app.core.config import get_settings
from app.services.document_loader import load_documents
from app.services.llm import build_embeddings_llm
from app.services.text_splitter import split_documents
from app.services.vector_store import build_vector_store
from app.utils.logging import setup_logging

logger = setup_logging()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    parser.add_argument(
        "--source",
        help="Re-index only this source file (deletes its existing chunks first).",
    )
    args = parser.parse_args()

    settings = get_settings()
    source = args.source

    documents = load_documents(settings.raw_docx_dir, settings.json_docs_dir)
    if not documents:
        logger.warning("No documents found in %s / %s", settings.raw_docx_dir, settings.json_docs_dir)
        return

    if source:
        documents = [doc for doc in documents if doc["metadata"].get("source") == source]
        if not documents:
            logger.warning("No documents loaded for source %s", source)
            return
        vector_store = build_vector_store(settings, build_embeddings_llm(settings))
        vector_store.delete(where={"source": source})
        logger.info("Deleted existing chunks for source %s", source)
    else:
        vector_store = build_vector_store(settings, build_embeddings_llm(settings))

    chunks = split_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        logger.warning("No chunks produced for source %s", source or "all documents")
        return

    vector_store.add_texts(
        texts=[chunk["page_content"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )
    logger.info(
        "Indexed %d chunks from %s into %s",
        len(chunks),
        source or f"{len(documents)} documents",
        settings.chroma_collection_name,
    )


if __name__ == "__main__":
    main()
