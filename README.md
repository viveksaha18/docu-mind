# DocuMind

DocuMind is a small, full-stack PDF question-answering app. Upload a text-based PDF, ask a question about it, and get an answer with source page references. The backend keeps the Gemini key; the browser only talks to the FastAPI API.

## Architecture

```text
React + Vite
    |  PDF upload / question
    v
FastAPI
    |-- PyMuPDF: extract text one page at a time
    |-- chunker: split text, keep page numbers, overlap neighboring chunks
    |-- Gemini embedding API: turn chunks and questions into vectors
    |-- FAISS: find the closest chunks for a question
    `-- Gemini generation API: answer from the retrieved excerpts
```

Each uploaded document gets its own in-memory FAISS index and a random document ID. Question requests include that ID, so retrieval only searches the selected PDF. Source excerpts and their page numbers travel together from extraction through the API response.

## How RAG works here

RAG means retrieval-augmented generation. Instead of asking a language model to answer from memory, DocuMind first retrieves relevant text from the uploaded PDF and then asks Gemini to answer using those passages.

1. **Extract:** PyMuPDF reads selectable text from each PDF page. Page numbering starts at 1. Scanned image-only documents are rejected because this MVP does not run OCR.
2. **Chunk:** The chunker normalizes whitespace and creates windows of 1,200 characters with 200 characters of overlap. Overlap helps keep a sentence or idea that crosses a boundary available to either neighboring chunk. Each chunk stores all page numbers it overlaps.
3. **Embed:** Gemini `gemini-embedding-001` converts chunks into 768-number vectors. Questions use the same model with the retrieval-query task type; document chunks use the retrieval-document task type.
4. **Search:** FAISS `IndexFlatIP` finds the four closest vectors. Vectors are normalized first, so inner product is cosine similarity. A best-match score below `0.35` returns the explicit unsupported-answer message without asking the generation model.
5. **Generate:** Gemini `gemini-3.8-flash` receives only the retrieved excerpts and an instruction not to use outside knowledge. The API returns the matching page numbers and excerpts alongside the answer.

The relevance cutoff and prompt reduce unsupported answers, but no generative model can guarantee perfect factuality. Always check the cited pages for important decisions.

## Requirements

- Python 3.10 or newer
- Node.js 18 or newer and npm
- A Gemini API key from Google AI Studio

## Run locally

Run these commands from the repository root. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
# Add your key to GEMINI_API_KEY in the root .env file first.
uvicorn backend.app.main:app --reload
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL printed in the terminal (normally `http://localhost:5173`). Vite proxies `/api` requests to FastAPI on port `8000`. FastAPI's interactive API reference is at `http://localhost:8000/docs`.

For macOS/Linux, activate the environment with `source .venv/bin/activate`, add your key to the root `.env` file, and start Uvicorn as above.

## Environment variables

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `GEMINI_API_KEY` | Yes, for upload and questions | None | Backend-only Gemini credential. Never add a real key to source control or a `VITE_` variable. |
| `GEMINI_EMBEDDING_MODEL` | No | `gemini-embedding-001` | Gemini embedding model name. |
| `GEMINI_CHAT_MODEL` | No | `gemini-3.8-flash` | Gemini answer-generation model name. |
| `FRONTEND_ORIGIN` | No | `http://localhost:5173` | Allowed browser origin for the backend CORS policy. |
| `VITE_API_BASE_URL` | No | Empty in development | Frontend API origin when frontend and backend are deployed separately. Set at Vite build time. |

The root `.env` is loaded by the backend and ignored by Git. Put your real local key there; never commit it. The root `.env.example` lists the same settings with a placeholder. Deployment environments should use their secret/configuration settings instead of uploading `.env`.

## Tests

The backend tests use generated PDFs, real local FAISS search, and a fake Gemini gateway, so they do not need a key or network access:

```powershell
python -m pytest backend/tests -q
```

Build the frontend with:

```powershell
npm --prefix frontend run build
```

## Deployment

### Render backend

The root `render.yaml` defines the FastAPI web service. In Render, select **New + > Blueprint** and connect this repository. The service runs from the repository root so the `backend` Python package imports correctly. Add `GEMINI_API_KEY` as a secret and set `FRONTEND_ORIGIN` to the exact deployed Vercel origin, including `https://` and without a trailing slash. Render supplies the service port.

### Vercel frontend

Import this repository into Vercel and set the project **Root Directory** to `frontend`. Vercel's build output is `dist/`; the matching settings are also in `frontend/vercel.json`. Add `VITE_API_BASE_URL` in Vercel's Production environment using the Render service origin, for example `https://documind-api.onrender.com`, with no trailing slash. Redeploy after setting it because Vite embeds this value at build time. A safe local example is in `frontend/.env.example`.

The frontend calls the Render API directly, so the Render `FRONTEND_ORIGIN` value must match the Vercel production domain. If you use a custom Vercel domain, use that domain instead. The Vercel preview domains are separate origins and will not be allowed unless you configure CORS for them too.

This is an MVP deployment shape, not a multi-user production architecture. FAISS indexes live in process memory and disappear when the backend restarts; multiple backend workers do not share indexes. Use persistent storage and a shared vector database before deploying for durable or multi-instance use. Enforce authentication and rate limits before exposing it publicly.

This is an MVP deployment shape, not a multi-user production architecture. FAISS indexes live in process memory and disappear when the backend restarts; multiple backend workers do not share indexes. Use persistent storage and a shared vector database before deploying for durable or multi-instance use. Enforce authentication and rate limits before exposing it publicly.

## Project layout

```text
backend/app/       FastAPI routes, PDF extraction, chunking, Gemini and FAISS
backend/tests/     Offline backend tests
frontend/src/      React application and styles
frontend/          Vite app, lockfile, Vercel config, and build output
render.yaml        Render FastAPI Blueprint
```