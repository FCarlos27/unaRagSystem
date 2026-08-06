from pathlib import Path

from PyPDF2 import PdfReader

from app.utils.logging import get_logger

logger = get_logger("document_loader")


def load_pdfs(directory: str) -> list[dict]:
    documents = []
    pdf_dir = Path(directory)
    if not pdf_dir.exists():
        logger.warning("PDF directory %s does not exist", directory)
        return documents

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        logger.info("Loading %s", pdf_path.name)
        try:
            reader = PdfReader(str(pdf_path))
            text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
            if text.strip():
                documents.append(
                    {
                        "page_content": text,
                        "metadata": {"source": pdf_path.name},
                    }
                )
        except Exception as exc:
            logger.error("Failed to parse %s: %s", pdf_path.name, exc)

    return documents
