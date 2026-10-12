import weakref
from types import SimpleNamespace

import numpy as np

from backend.app.chunking import TextChunk
from backend.app.rag import (
    EMBEDDING_DIMENSIONS,
    UNSUPPORTED_ANSWER,
    DocumentIndex,
    GeminiGateway,
    answer_question,
)


class FakeGateway:
    def __init__(self, question_dimension: int = 0) -> None:
        self.question_dimension = question_dimension
        self.prompt = ""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), EMBEDDING_DIMENSIONS), dtype=np.float32)
        vectors[0, 0] = 1
        if len(texts) > 1:
            vectors[1, 1] = 1
        return vectors

    def embed_question(self, question: str) -> np.ndarray:
        vector = np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32)
        vector[self.question_dimension] = 1
        return vector

    def generate_answer(self, question: str, context: str) -> str:
        self.prompt = context
        return "The contract renews annually (p. 5)."


def make_document_index(gateway: FakeGateway) -> DocumentIndex:
    chunks = [
        TextChunk("The contract renews annually.", (5,)),
        TextChunk("Invoices are issued monthly.", (7,)),
    ]
    return DocumentIndex.from_chunks(chunks, gateway)


def test_retrieval_supplies_relevant_context_and_page_metadata() -> None:
    gateway = FakeGateway()
    index = make_document_index(gateway)

    answer, matches = answer_question("When does it renew?", index, gateway)

    assert answer == "The contract renews annually (p. 5)."
    assert matches[0].page_numbers == (5,)
    assert "The contract renews annually." in gateway.prompt


def test_low_relevance_returns_explicit_fallback_without_calling_answer_model() -> None:
    gateway = FakeGateway(question_dimension=2)
    index = make_document_index(gateway)

    answer, matches = answer_question("Who won the tournament?", index, gateway)

    assert answer == UNSUPPORTED_ANSWER
    assert matches == []
    assert gateway.prompt == ""


def test_generate_answer_keeps_client_alive_for_request(monkeypatch) -> None:
    client_reference = None

    class Models:
        def generate_content(self, **kwargs):
            assert client_reference() is not None
            return SimpleNamespace(text="Grounded answer.")

    class Client:
        def __init__(self, api_key: str) -> None:
            nonlocal client_reference
            client_reference = weakref.ref(self)
            self.models = Models()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("backend.app.rag.genai.Client", Client)

    assert GeminiGateway().generate_answer("Question?", "Context") == "Grounded answer."