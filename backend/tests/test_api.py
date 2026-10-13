import asyncio
import fitz
import threading
import time

from google.genai.errors import ClientError
from httpx import ASGITransport, AsyncClient
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


def test_upload_embedding_does_not_block_health_endpoint(monkeypatch) -> None:
    embedding_started = threading.Event()
    release_embedding = threading.Event()

    class SlowGateway(FakeGateway):
        def embed_documents(self, texts: list[str]) -> np.ndarray:
            embedding_started.set()
            release_embedding.wait(timeout=0.5)
            return super().embed_documents(texts)

    async def upload_and_check_health() -> tuple[int, int, float]:
        async with AsyncClient(
            transport=ASGITransport(app=main.app), base_url="http://test"
        ) as client:
            upload_started = time.monotonic()
            upload_task = asyncio.create_task(
                client.post(
                    "/api/documents",
                    files={"file": ("policy.pdf", make_pdf(), "application/pdf")},
                )
            )
            await asyncio.wait_for(asyncio.to_thread(embedding_started.wait, 1), timeout=1.5)
            health_response = await client.get("/api/health")
            health_elapsed = time.monotonic() - upload_started
            release_embedding.set()
            upload_response = await upload_task
        return upload_response.status_code, health_response.status_code, health_elapsed

    monkeypatch.setattr(main, "gemini_gateway", SlowGateway())
    release_timer = threading.Timer(0.5, release_embedding.set)
    release_timer.daemon = True
    release_timer.start()
    try:
        upload_status, health_status, health_elapsed = asyncio.run(upload_and_check_health())
    finally:
        release_embedding.set()
        release_timer.cancel()

    assert upload_status == 200
    assert health_status == 200
    assert health_elapsed < 0.4


def test_upload_rejects_non_pdf() -> None:
    response = TestClient(main.app).post(
        "/api/documents", files={"file": ("notes.txt", b"plain text", "text/plain")}
    )
    assert response.status_code == 415


def test_upload_rejects_malformed_pdf() -> None:
    response = TestClient(main.app).post(
        "/api/documents", files={"file": ("broken.pdf", b"%PDF-1.7\nmalformed", "application/pdf")}
    )
    assert response.status_code == 422


def test_upload_rejects_files_over_size_limit(monkeypatch) -> None:
    monkeypatch.setattr(main, "MAX_PDF_BYTES", 16)
    response = TestClient(main.app).post(
        "/api/documents", files={"file": ("large.pdf", b"x" * 17, "application/pdf")}
    )
    assert response.status_code == 413


def test_question_handles_gemini_quota_exhaustion_gracefully(monkeypatch) -> None:
    class QuotaGateway(FakeGateway):
        def generate_answer(self, question: str, context: str) -> str:
            raise ClientError(429, {"error": {"message": "RESOURCE_EXHAUSTED"}}, None)

    monkeypatch.setattr(main, "gemini_gateway", QuotaGateway())
    client = TestClient(main.app)

    upload = client.post(
        "/api/documents",
        files={"file": ("policy.pdf", make_pdf(), "application/pdf")},
    )
    assert upload.status_code == 200

    response = client.post(
        f"/api/documents/{upload.json()['document_id']}/questions",
        json={"question": "What does it define?"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == main.GEMINI_QUOTA_MESSAGE
    assert response.json()["sources"] == []


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