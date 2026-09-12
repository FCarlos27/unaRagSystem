from pathlib import Path

from docx import Document

from app.utils.logging import get_logger

logger = get_logger("document_loader")


def _load_docx(docx_path: Path) -> str:
    """Extract paragraph and table text from a .docx file."""
    doc = Document(str(docx_path))
    text_parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            text_parts.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                text_parts.append("\t".join(cells))
    return "\n".join(text_parts)


def load_docx_files(directory: str) -> list[dict]:
    """Load every .docx file in a directory into raw text documents."""
    documents = []
    docx_dir = Path(directory)
    if not docx_dir.exists():
        logger.warning("DOCX directory %s does not exist", directory)
        return documents

    for docx_path in sorted(docx_dir.glob("*.docx")):
        if docx_path.name.startswith("~$"):
            continue
        logger.info("Loading %s", docx_path.name)
        try:
            text = _load_docx(docx_path)
            if text.strip():
                documents.append(
                    {
                        "page_content": text,
                        "metadata": {"source": docx_path.name},
                    }
                )
        except Exception as exc:
            logger.error("Failed to parse %s: %s", docx_path.name, exc)

    return documents


def load_markdown_files(directory: str) -> list[dict]:
    """Load every .md file in a directory into raw text documents."""
    documents = []
    md_dir = Path(directory)
    if not md_dir.exists():
        return documents

    for md_path in sorted(md_dir.glob("*.md")):
        logger.info("Loading %s", md_path.name)
        try:
            text = md_path.read_text(encoding="utf-8")
            if text.strip():
                documents.append(
                    {
                        "page_content": text,
                        "metadata": {
                            "source": md_path.name,
                            "format": "markdown",
                        },
                    }
                )
        except Exception as exc:
            logger.error("Failed to parse %s: %s", md_path.name, exc)

    return documents


def load_documents(
    docx_dir: str, md_dir: str | None = None
) -> list[dict]:
    """Load DOCX and Markdown documents from the given directories."""
    return load_docx_files(docx_dir) + load_markdown_files(md_dir or docx_dir)
