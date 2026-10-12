"""Text chunking that keeps source page metadata alongside each chunk."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    text: str
    page_numbers: tuple[int, ...]


def chunk_pages(
    pages: list[tuple[int, str]], chunk_size: int = 1200, overlap: int = 200
) -> list[TextChunk]:
    """Split page text into character windows and retain every contributing page."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    text_parts: list[str] = []
    page_ranges: list[tuple[int, int, int]] = []
    cursor = 0

    for page_number, page_text in pages:
        normalized_text = " ".join(page_text.split())
        if not normalized_text:
            continue
        if text_parts:
            text_parts.append("\n")
            cursor += 1
        start = cursor
        text_parts.append(normalized_text)
        cursor += len(normalized_text)
        page_ranges.append((start, cursor, page_number))

    full_text = "".join(text_parts)
    if not full_text:
        return []

    chunks: list[TextChunk] = []
    step = chunk_size - overlap
    for start in range(0, len(full_text), step):
        end = min(start + chunk_size, len(full_text))
        chunk_text = full_text[start:end].strip()
        if not chunk_text:
            continue
        source_pages = tuple(
            page_number
            for page_start, page_end, page_number in page_ranges
            if page_start < end and page_end > start
        )
        chunks.append(TextChunk(text=chunk_text, page_numbers=source_pages))
        if end == len(full_text):
            break

    return chunks