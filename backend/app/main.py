"""FastAPI endpoints for uploading PDFs and asking document-scoped questions."""

import logging
import os
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from backend.app.chunking import chunk_pages
from backend.app.pdf_processing import PDFProcessingError, extract_pdf_pages
from backend.app.rag import (
    UNSUPPORTED_ANSWER,
    DocumentIndex,
    GeminiGateway,
    answer_question,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)
MAX_PDF_BYTES = 20 * 1024 * 1024
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

app = FastAPI(title="DocuMind API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

gemini_gateway = GeminiGateway()
document_indexes: dict[str, DocumentIndex] = {}


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    page_count: int
    chunk_count: int


class QuestionRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 2:
            raise ValueError("Enter a question with at least two non-space characters.")
        return cleaned


class SourceResponse(BaseModel):
    page_numbers: list[int]
    excerpt: str
    relevance: float


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/documents", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    filename = file.filename or ""
    if file.content_type != "application/pdf" and not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Upload a PDF file.")

    pdf_bytes = await file.read(MAX_PDF_BYTES + 1)
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF files must be 20 MB or smaller.")

    try:
        pages = extract_pdf_pages(pdf_bytes)
    except PDFProcessingError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    try:
        index = DocumentIndex.from_chunks(chunks, gemini_gateway)
    except RuntimeError as error:
        logger.exception("Could not create embeddings for uploaded document")
        raise HTTPException(
            status_code=503,
            detail="Could not create embeddings. Check the backend Gemini API key and availability.",
        ) from error
    except Exception as error:
        logger.exception("Gemini embedding request failed")
        raise HTTPException(status_code=502, detail="Gemini could not process this PDF.") from error

    document_id = uuid4().hex
    document_indexes[document_id] = index
    return UploadResponse(
        document_id=document_id,
        filename=filename or "document.pdf",
        page_count=len(pages),
        chunk_count=len(chunks),
    )


@app.post("/api/documents/{document_id}/questions", response_model=AnswerResponse)
def ask_question(document_id: str, request: QuestionRequest) -> AnswerResponse:
    document_index = document_indexes.get(document_id)
    if document_index is None:
        raise HTTPException(status_code=404, detail="Document not found. Upload it again to ask questions.")

    try:
        answer, matches = answer_question(request.question, document_index, gemini_gateway)
    except RuntimeError as error:
        logger.exception("Gemini could not answer document question")
        raise HTTPException(
            status_code=503,
            detail="Gemini is unavailable. Check the backend API key and try again.",
        ) from error
    except Exception as error:
        logger.exception("Document question processing failed")
        raise HTTPException(status_code=502, detail="Could not process the question.") from error

    if answer == UNSUPPORTED_ANSWER:
        return AnswerResponse(answer=answer, sources=[])

    sources = [
        SourceResponse(
            page_numbers=list(match.page_numbers),
            excerpt=match.text[:280],
            relevance=round(match.score, 3),
        )
        for match in matches
    ]
    return AnswerResponse(answer=answer, sources=sources)