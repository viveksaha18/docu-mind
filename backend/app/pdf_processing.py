"""PDF validation and page-preserving text extraction."""

import pymupdf as fitz


class PDFProcessingError(ValueError):
    """Raised when a PDF cannot provide usable text."""


def extract_pdf_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    if not pdf_bytes.startswith(b"%PDF"):
        raise PDFProcessingError("The uploaded file is not a valid PDF.")

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            pages = [
                (page_number, page.get_text("text", sort=True).strip())
                for page_number, page in enumerate(document, start=1)
            ]
    except (fitz.FileDataError, RuntimeError) as error:
        raise PDFProcessingError("The uploaded PDF could not be opened.") from error

    if not any(text for _, text in pages):
        raise PDFProcessingError(
            "No selectable text was found. Scanned PDFs need OCR, which this MVP does not include."
        )
    return pages