import pymupdf as fitz
import pytest

from backend.app.pdf_processing import PDFProcessingError, extract_pdf_pages


def make_pdf(*page_texts: str) -> bytes:
    document = fitz.open()
    for page_text in page_texts:
        page = document.new_page()
        page.insert_text((72, 72), page_text)
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def test_extracts_text_with_one_based_page_numbers() -> None:
    assert extract_pdf_pages(make_pdf("First page", "Second page")) == [
        (1, "First page"),
        (2, "Second page"),
    ]


def test_rejects_invalid_or_textless_pdf() -> None:
    with pytest.raises(PDFProcessingError, match="not a valid PDF"):
        extract_pdf_pages(b"not a pdf")

    with pytest.raises(PDFProcessingError, match="No selectable text"):
        extract_pdf_pages(make_pdf(""))