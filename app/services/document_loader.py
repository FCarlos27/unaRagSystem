import json
from io import BytesIO
from pathlib import Path

import fitz
import pytesseract
from PIL import Image
from PyPDF2 import PdfReader

from app.utils.logging import get_logger

logger = get_logger("document_loader")

OCR_LANGUAGE = "spa"
MIN_OCR_IMAGE_SIZE = 256


def _ocr_image_bytes(image_bytes: bytes) -> str:
    """Run Tesseract OCR on raw image bytes; returns text or '' on failure."""
    try:
        image = Image.open(BytesIO(image_bytes))
        if min(image.size) < MIN_OCR_IMAGE_SIZE:
            logger.debug("Skipping small image %s", image.size)
            return ""
        return pytesseract.image_to_string(image, lang=OCR_LANGUAGE).strip()
    except Exception as exc:
        logger.warning("OCR failed for image: %s", exc)
        return ""


def _extract_page_images(pdf_doc: fitz.Document, page, page_number: int) -> str:
    """Extract and OCR all images embedded in a single PDF page."""
    ocr_parts = []
    for img_index, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        try:
            base_image = pdf_doc.extract_image(xref)
            ocr_text = _ocr_image_bytes(base_image["image"])
            if ocr_text:
                ocr_parts.append(
                    f"[Texto extraído de la imagen {img_index + 1} "
                    f"en la página {page_number}]:\n{ocr_text}"
                )
        except Exception as exc:
            logger.warning("Failed to extract image xref %s: %s", xref, exc)
    return "\n\n".join(ocr_parts)


def _load_pdf(pdf_path: Path) -> str:
    """Read a PDF's text layer plus OCR'd text from its embedded images."""
    reader = PdfReader(str(pdf_path))
    pages_text = [page.extract_text() or "" for page in reader.pages]

    pdf_doc = fitz.open(str(pdf_path))
    try:
        for page_number, page in enumerate(pdf_doc, start=1):
            ocr_text = _extract_page_images(pdf_doc, page, page_number)
            if ocr_text:
                pages_text[page_number - 1] += f"\n\n{ocr_text}"
    finally:
        pdf_doc.close()

    return "\n".join(pages_text)


def load_pdfs(directory: str) -> list[dict]:
    """Load every PDF in a directory into raw text documents."""
    documents = []
    pdf_dir = Path(directory)
    if not pdf_dir.exists():
        logger.warning("PDF directory %s does not exist", directory)
        return documents

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        logger.info("Loading %s", pdf_path.name)
        try:
            text = _load_pdf(pdf_path)
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


def _sanitize_metadata(entry: dict) -> dict:
    """Keep only Chroma-compatible scalar metadata (drop None/dicts, join lists)."""
    meta = {}
    for key, value in entry.items():
        if key == "content_chunk":
            continue
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(item) for item in value)
        if isinstance(value, (str, int, float, bool)):
            meta[key] = value
    return meta


def _entry_lists(data) -> list[list[dict]]:
    """Find the list-of-dicts arrays in a JSON document (tables/registries)."""
    known_keys = ("directorio", "directorio_contactos", "cuentas_bancarias", "registros")
    for key in known_keys:
        value = data.get(key)
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return [value]
    lists = [
        value
        for value in data.values()
        if isinstance(value, list) and value and isinstance(value[0], dict)
    ]
    return lists if lists else [data.get("directorio", [])]


def _build_content_chunk(entry: dict) -> str:
    """Generate readable text from a JSON row when no content_chunk is provided."""
    lines = []
    for key, value in entry.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(item) for item in value)
        if not isinstance(value, (str, int, float, bool)):
            continue
        label = key.replace("_", " ").title()
        lines.append(f"{label}: {value}")
    return "\n".join(lines)


def load_json_files(directory: str) -> list[dict]:
    """Load structured JSON entries as prechunked documents."""
    documents = []
    json_dir = Path(directory)
    if not json_dir.exists():
        return documents

    for json_path in sorted(json_dir.glob("*.json")):
        logger.info("Loading %s", json_path.name)
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            for entries in _entry_lists(data):
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    content = entry.get("content_chunk") or _build_content_chunk(entry)
                    if not content:
                        logger.warning("Empty entry in %s", json_path.name)
                        continue
                    documents.append(
                        {
                            "page_content": content,
                            "metadata": {
                                "source": json_path.name,
                                "prechunked": True,
                                **_sanitize_metadata(entry),
                            },
                        }
                    )
        except Exception as exc:
            logger.error("Failed to parse %s: %s", json_path.name, exc)

    return documents


def load_documents(
    pdfs_directory: str, json_directory: str | None = None
) -> list[dict]:
    """Load all PDFs and JSON documents from the given directories."""
    return load_pdfs(pdfs_directory) + load_json_files(json_directory or pdfs_directory)
