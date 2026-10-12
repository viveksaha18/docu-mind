import fitz
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.rag import EMBEDDING_DIMENSIONS
import numpy as np


class FakeGateway:
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), EMBEDDING_DIMENSIONS), dtype=np.float32)
        vectors[:, 0] = 1
        return vectors

    def embed_question(self, question: str) -> np.ndarray:
        vector = np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32)
        vector[0] = 1
        return vector

    def generate_answer(self, question: str, context: str) -> str:
        return "The document defines the policy (p. 1)."


def make_pdf() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "The document defines the policy.")
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def test_upload_then_ask_returns_answer_and_source_page(monkeypatch) -> None:
    monkeypatch.setattr(main, "gemini_gateway", FakeGateway())
    client = TestClient(main.app)

    upload = client.post(
        "/api/documents",
        files={"file": ("policy.pdf", make_pdf(), "application/pdf")},
    )
    assert upload.status_code == 200
    assert upload.json()["page_count"] == 1

    answer = client.post(
        f"/api/documents/{upload.json()['document_id']}/questions",
        json={"question": "What does it define?"},
    )
    assert answer.status_code == 200
    assert answer.json()["answer"] == "The document defines the policy (p. 1)."
    assert answer.json()["sources"][0]["page_numbers"] == [1]


def test_upload_rejects_non_pdf() -> None:
    response = TestClient(main.app).post(
        "/api/documents", files={"file": ("notes.txt", b"plain text", "text/plain")}
    )
    assert response.status_code == 415


def test_question_requires_text_and_known_document() -> None:
    client = TestClient(main.app)

    blank_question = client.post(
        "/api/documents/unknown/questions", json={"question": "  "}
    )
    unknown_document = client.post(
        "/api/documents/unknown/questions", json={"question": "Explain this"}
    )

    assert blank_question.status_code == 422
    assert unknown_document.status_code == 404