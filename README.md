# DocuMind

DocuMind is a small, full-stack PDF question-answering app. Upload a text-based PDF, ask a question about it, and get an answer with source page references. The backend uses Ollama for embeddings and answer generation; the browser only talks to the FastAPI API. Ollama chat can run locally or use a remote Ollama endpoint.

## Architecture

```text
React + Vite
    |  PDF upload / question
    v
FastAPI
    |-- PyMuPDF: extract text one page at a time
    |-- chunker: split text, keep page numbers, overlap neighboring chunks
    |-- Ollama embedding model: turn chunks and questions into vectors
    |-- FAISS: find the closest chunks for a question
    `-- Ollama chat model: answer from the retrieved excerpts
```

Each uploaded document gets its own in-memory FAISS index and a random document ID. Question requests include that ID, so retrieval only searches the selected PDF. Source excerpts and their page numbers travel together from extraction through the API response.

## How RAG works here

RAG means retrieval-augmented generation. Instead of asking a language model to answer from memory, DocuMind first retrieves relevant text from the uploaded PDF and then asks the configured Ollama chat model to answer using those passages.

1. **Extract:** PyMuPDF reads selectable text from each PDF page. Page numbering starts at 1. Scanned image-only documents are rejected because this MVP does not run OCR.
2. **Chunk:** The chunker normalizes whitespace and creates windows of 1,200 characters with 200 characters of overlap. Overlap helps keep a sentence or idea that crosses a boundary available to either neighboring chunk. Each chunk stores all page numbers it overlaps.
3. **Embed:** Ollama's `nomic-embed-text` model converts chunks and questions into 768-number vectors by default.
4. **Search:** FAISS `IndexFlatIP` finds the four closest vectors. Vectors are normalized first, so inner product is cosine similarity. A best-match score below `0.35` returns the explicit unsupported-answer message without asking the generation model.
5. **Generate:** The Ollama chat model (default `gpt-oss`) receives only the retrieved excerpts and an instruction not to use outside knowledge. The API returns the matching page numbers and excerpts alongside the answer.

The relevance cutoff and prompt reduce unsupported answers, but no generative model can guarantee perfect factuality. Always check the cited pages for important decisions.

## Requirements

- Python 3.10 or newer
- Node.js 18 or newer and npm
- Ollama installed and running, with the configured embedding and chat models available

## Run locally

Run these commands from the repository root in Windows PowerShell. Create a root `.env` file with Ollama configuration so the backend selects Ollama instead of its Gemini fallback:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_CHAT_MODEL=gpt-oss
```

Pull the models if they are not already available in Ollama:

```powershell
ollama pull nomic-embed-text
ollama pull gpt-oss
```

Then install and start the backend:

```powershell
py -3.12 -m venv backend/.venv
.\backend\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
Push-Location backend
python -m uvicorn app.main:app --reload
Pop-Location
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL printed in the terminal (normally `http://localhost:5173`). Vite proxies `/api` requests to FastAPI on port `8000`. FastAPI's interactive API reference is at `http://localhost:8000/docs`.

For macOS/Linux, create and activate `backend/.venv`, configure the root `.env` file as above, then run `python -m uvicorn app.main:app --app-dir backend --reload` from the repository root.

By default, embeddings and chat use the Ollama server at `OLLAMA_BASE_URL`. To use a remote Ollama chat endpoint while keeping embeddings local, set `OLLAMA_CHAT_BASE_URL` to that endpoint and set `OLLAMA_API_KEY` for authentication. The API key is used for remote chat requests. Ensure the remote endpoint is reachable from the machine running the backend. The frontend connects to the backend at `http://localhost:8000`; FastAPI's interactive API reference is at `http://localhost:8000/docs`.

## Environment variables

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `OLLAMA_BASE_URL` | No, but set it to select Ollama | `http://localhost:11434` | Ollama endpoint used for embeddings and, unless overridden, chat. |
| `OLLAMA_EMBEDDING_MODEL` | No | `nomic-embed-text` | Ollama model used to embed PDF chunks and questions. Must produce 768-dimensional vectors. |
| `OLLAMA_CHAT_MODEL` | No | `gpt-oss` | Ollama model used to generate answers. `OLLAMA_MODEL` is also accepted as a legacy answer-model setting and selects Ollama mode. |
| `OLLAMA_CHAT_BASE_URL` | No | `OLLAMA_BASE_URL` | Optional separate endpoint for Ollama chat requests. |
| `OLLAMA_API_KEY` | Only for remote chat | None | Bearer credential for a non-local Ollama chat endpoint. Keep it out of source control and frontend variables. |
| `FRONTEND_ORIGIN` | No | `http://localhost:5173` | Allowed browser origin for the backend CORS policy. |
| `VITE_API_BASE_URL` | No | Empty in development | Frontend API origin when frontend and backend are deployed separately. Set at Vite build time. |

The root `.env` is loaded by the backend and ignored by Git. Do not commit credentials. If no Ollama selector variable (`OLLAMA_BASE_URL`, `OLLAMA_MODEL`, or `OLLAMA_EMBEDDING_MODEL`) is set, the backend falls back to Gemini and requires `GEMINI_API_KEY`.

## Tests

The backend tests use generated PDFs, real local FAISS search, and a fake model gateway, so they do not need Ollama, a Gemini key, or network access:

```powershell
.\backend\.venv\Scripts\python.exe -m pytest backend/tests -q
```

Build the frontend with the Render API base URL configured:

```powershell
$env:VITE_API_BASE_URL = "https://your-documind-api.onrender.com"
npm --prefix frontend run build
Remove-Item Env:VITE_API_BASE_URL
```

## Deployment

### Render backend

The root `render.yaml` defines the FastAPI web service with `backend` as its root directory, but its environment variables are still configured for Gemini. To use Ollama, set the required `OLLAMA_*` variables in Render and ensure the backend can reach the configured Ollama endpoints; `localhost` in Render refers to the Render service itself, not your computer. The checked-in Blueprint must also be updated to manage those variables if you deploy through it. Alternatively, configure the service manually with Root Directory `backend`, Build Command `pip install -r requirements.txt`, and Start Command `uvicorn app.main:app --host 0.0.0.0 --port $PORT`. Set `FRONTEND_ORIGIN` to the exact deployed Vercel origin, including `https://` and without a trailing slash. Render supplies the port.

### Vercel frontend

Import this repository into Vercel and set the project **Root Directory** to `frontend`. Vercel's build output is `dist/`; the matching settings are also in `frontend/vercel.json`. Add `VITE_API_BASE_URL` in Vercel's Production environment using the Render service origin, for example `https://documind-api.onrender.com`, with no trailing slash. Redeploy after setting it because Vite embeds this value at build time. Production builds fail if the variable is missing.

The frontend calls the Render API directly, so the Render `FRONTEND_ORIGIN` value must match the Vercel production domain. If you use a custom Vercel domain, use that domain instead. The Vercel preview domains are separate origins and will not be allowed unless you configure CORS for them too.

This is an MVP deployment shape, not a multi-user production architecture. FAISS indexes live in process memory and disappear when the backend restarts; multiple backend workers do not share indexes. Use persistent storage and a shared vector database before deploying for durable or multi-instance use. Enforce authentication and rate limits before exposing it publicly.

## Project layout

```text
backend/app/       FastAPI routes, PDF extraction, chunking, Ollama/Gemini and FAISS
backend/tests/     Offline backend tests
frontend/src/      React application and styles
frontend/          Vite app, lockfile, Vercel config, and build output
render.yaml        Render FastAPI Blueprint
```