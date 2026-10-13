"""FastAPI endpoints for uploading PDFs and asking document-scoped questions."""

import logging
import os
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google.genai.errors import ClientError
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from .chunking import chunk_pages
from .pdf_processing import PDFProcessingError, extract_pdf_pages
from .rag import (
    UNSUPPORTED_ANSWER,
    DocumentIndex,
    GeminiGateway,
    answer_question,
    get_default_gateway,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)
MAX_PDF_BYTES = 20 * 1024 * 1024
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
GEMINI_QUOTA_MESSAGE = (
    "The answer service is temporarily unavailable because the Gemini API quota has been reached. "
    "Please try again later or check your API billing."
)

app = FastAPI(title="DocuMind API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

gemini_gateway = get_default_gateway()
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
        pages = await run_in_threadpool(extract_pdf_pages, pdf_bytes)
    except PDFProcessingError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    chunks = await run_in_threadpool(
        chunk_pages, pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP
    )
    try:
        index = await run_in_threadpool(
            DocumentIndex.from_chunks, chunks, gemini_gateway
        )
    except ClientError as error:
        if getattr(error, "status_code", None) == 429 or "RESOURCE_EXHAUSTED" in str(error).upper():
            raise HTTPException(
                status_code=429,
                detail="Gemini API quota has been reached. Please try again later or add billing.",
            ) from error
        logger.exception("Gemini embedding request failed")
        raise HTTPException(status_code=502, detail="Gemini could not process this PDF.") from error
    except RuntimeError as error:
        logger.exception("Could not create embeddings for uploaded document")
        raise HTTPException(
            status_code=503,
            detail="Could not create embeddings. Check the configured embedding model and provider availability.",
        ) from error
    except Exception as error:
        logger.exception("Embedding provider could not process uploaded PDF")
        raise HTTPException(status_code=502, detail="The embedding provider could not process this PDF.") from error

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
    except ClientError as error:
        if getattr(error, "status_code", None) == 429 or "RESOURCE_EXHAUSTED" in str(error).upper():
            return AnswerResponse(answer=GEMINI_QUOTA_MESSAGE, sources=[])
        logger.exception("Configured model provider could not answer document question")
        raise HTTPException(
            status_code=503,
            detail="The configured model provider is unavailable. Check its URL, API key, and model name.",
        ) from error
    except RuntimeError as error:
        logger.exception("Configured model provider could not answer document question")
        raise HTTPException(
            status_code=503,
            detail="The configured model provider is unavailable. Check its URL, API key, and model name.",
        ) from error
    except Exception as error:
        logger.exception("Document question processing failed")
        raise HTTPException(status_code=502, detail="Could not process the question. Check the backend logs for details.") from error

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