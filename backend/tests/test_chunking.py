import pytest

from backend.app.chunking import chunk_pages


def test_chunks_overlap_and_keep_page_metadata() -> None:
    chunks = chunk_pages([(1, "abcdefghijklmnopqrst")], chunk_size=10, overlap=3)

    assert chunks[0].text == "abcdefghij"
    assert chunks[1].text.startswith("hij")
    assert all(chunk.page_numbers == (1,) for chunk in chunks)


def test_chunk_spanning_pages_lists_both_sources() -> None:
    chunks = chunk_pages([(4, "first page"), (5, "second page")], chunk_size=15, overlap=2)

    assert chunks[0].page_numbers == (4, 5)


def test_empty_pages_produce_no_chunks() -> None:
    assert chunk_pages([(1, " "), (2, "")]) == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap"), [(0, 0), (10, -1), (10, 10)]
)
def test_invalid_chunk_configuration_is_rejected(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_pages([(1, "text")], chunk_size=chunk_size, overlap=overlap)