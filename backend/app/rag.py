"""Gemini embeddings, FAISS retrieval, and context-grounded answers."""

import os
from dataclasses import dataclass

import faiss
import numpy as np
from google import genai
from google.genai import types

from backend.app.chunking import TextChunk

UNSUPPORTED_ANSWER = "I couldn't find an answer to that in the uploaded document."
EMBEDDING_DIMENSIONS = 768


class GeminiGateway:
    """Keep all Gemini API access and configuration on the backend."""

    def __init__(self) -> None:
        self.embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        self.answer_model = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash")

    def _client(self) -> genai.Client:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured on the backend.")
        return genai.Client(api_key=api_key)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        client = self._client()
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), 100):
            response = client.models.embed_content(
                model=self.embedding_model,
                contents=texts[offset : offset + 100],
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=EMBEDDING_DIMENSIONS,
                ),
            )
            vectors.extend(embedding.values for embedding in response.embeddings or [])
        if len(vectors) != len(texts):
            raise RuntimeError("Gemini returned an unexpected number of document embeddings.")
        return np.asarray(vectors, dtype=np.float32)

    def embed_question(self, question: str) -> np.ndarray:
        client = self._client()
        response = client.models.embed_content(
            model=self.embedding_model,
            contents=question,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )
        if not response.embeddings:
            raise RuntimeError("Gemini did not return a question embedding.")
        return np.asarray(response.embeddings[0].values, dtype=np.float32)

    def generate_answer(self, question: str, context: str) -> str:
        prompt = f"""You answer questions using only the document excerpts below.
If the excerpts do not contain enough evidence, reply exactly:
{UNSUPPORTED_ANSWER}
Do not use outside knowledge. Do not invent details. Cite supporting page numbers inline
using the format (p. 3) or (pp. 3, 4). Keep the answer concise.

Document excerpts:
{context}

Question: {question}
Answer:"""
        client = self._client()
        response = client.models.generate_content(
            model=self.answer_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1),
        )
        answer = (response.text or "").strip()
        return answer or UNSUPPORTED_ANSWER


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    page_numbers: tuple[int, ...]
    score: float


class DocumentIndex:
    def __init__(self, chunks: list[TextChunk], vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or len(vectors) != len(chunks):
            raise ValueError("Each text chunk must have exactly one embedding vector.")
        if vectors.shape[1] != EMBEDDING_DIMENSIONS:
            raise ValueError("Embedding dimensions do not match the configured Gemini model output.")

        self.chunks = chunks
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
        normalized_vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        faiss.normalize_L2(normalized_vectors)
        self.index.add(normalized_vectors)

    @classmethod
    def from_chunks(cls, chunks: list[TextChunk], gateway: GeminiGateway) -> "DocumentIndex":
        vectors = gateway.embed_documents([chunk.text for chunk in chunks])
        return cls(chunks, vectors)

    def search(
        self, question_vector: np.ndarray, limit: int = 4
    ) -> list[RetrievedChunk]:
        if not self.chunks:
            return []
        query = np.ascontiguousarray(question_vector.reshape(1, -1), dtype=np.float32)
        if query.shape[1] != EMBEDDING_DIMENSIONS:
            raise ValueError("Question embedding dimensions do not match the document index.")
        faiss.normalize_L2(query)
        scores, positions = self.index.search(query, min(limit, len(self.chunks)))
        return [
            RetrievedChunk(
                text=self.chunks[position].text,
                page_numbers=self.chunks[position].page_numbers,
                score=float(score),
            )
            for score, position in zip(scores[0], positions[0], strict=True)
            if position >= 0
        ]


def answer_question(
    question: str,
    document_index: DocumentIndex,
    gateway: GeminiGateway,
    relevance_threshold: float = 0.35,
) -> tuple[str, list[RetrievedChunk]]:
    matches = document_index.search(gateway.embed_question(question))
    if not matches or matches[0].score < relevance_threshold:
        return UNSUPPORTED_ANSWER, []

    context = "\n\n".join(
        f"[Pages {', '.join(map(str, match.page_numbers))}]\n{match.text}"
        for match in matches
    )
    return gateway.generate_answer(question, context), matches