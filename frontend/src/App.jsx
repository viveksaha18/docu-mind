import { useEffect, useRef, useState } from "react";
import {
  ArrowDownToLine,
  ArrowUp,
  BookOpenText,
  Check,
  CircleAlert,
  FileText,
  Plus,
  Sparkles,
  Upload,
  X,
} from "lucide-react";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
const MAX_FILE_SIZE = 20 * 1024 * 1024;

async function readApiResponse(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail ?? "Something went wrong. Please try again.");
  }
  return payload;
}

function pageLabel(pages) {
  return pages.map((page) => `p. ${page}`).join(", ");
}

function App() {
  const [documentInfo, setDocumentInfo] = useState(null);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [isAsking, setIsAsking] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState("");
  const [apiStatus, setApiStatus] = useState("checking");
  const [selectedSource, setSelectedSource] = useState(null);
  const fileInputRef = useRef(null);
  const conversationEndRef = useRef(null);
  const latestSources = messages.at(-1)?.sources ?? [];

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/health`)
      .then((response) => {
        if (!response.ok) throw new Error("API unavailable");
        setApiStatus("online");
      })
      .catch(() => setApiStatus("offline"));
  }, []);

  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isAsking]);

  async function uploadFile(file) {
    if (!file) return;
    setError("");
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Choose a PDF file to continue.");
      return;
    }
    if (file.size > MAX_FILE_SIZE) {
      setError("This PDF is over 20 MB. Choose a smaller file.");
      return;
    }

    setIsUploading(true);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch(`${API_BASE_URL}/api/documents`, {
        method: "POST",
        body: formData,
      });
      const uploadedDocument = await readApiResponse(response);
      setDocumentInfo(uploadedDocument);
      setMessages([]);
      setSelectedSource(null);
    } catch (uploadError) {
      setError(uploadError.message);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function submitQuestion(event) {
    event.preventDefault();
    const cleanedQuestion = question.trim();
    if (!cleanedQuestion || !documentInfo || isAsking) return;

    setQuestion("");
    setError("");
    setIsAsking(true);
    setMessages((current) => [...current, { question: cleanedQuestion, answer: null, sources: [] }]);
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/documents/${documentInfo.document_id}/questions`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: cleanedQuestion }),
        },
      );
      const answer = await readApiResponse(response);
      setMessages((current) =>
        current.map((message, index) =>
          index === current.length - 1
            ? { ...message, answer: answer.answer, sources: answer.sources }
            : message,
        ),
      );
    } catch (questionError) {
      setError(questionError.message);
      setMessages((current) => current.slice(0, -1));
      setQuestion(cleanedQuestion);
    } finally {
      setIsAsking(false);
    }
  }

  function startNewDocument() {
    setDocumentInfo(null);
    setMessages([]);
    setError("");
    setSelectedSource(null);
  }

  function handleDrop(event) {
    event.preventDefault();
    setIsDragging(false);
    uploadFile(event.dataTransfer.files[0]);
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#workspace" aria-label="DocuMind home">
          <span className="brand-mark"><BookOpenText size={19} strokeWidth={2.2} /></span>
          <span>documind<span className="brand-period">.</span></span>
        </a>

        <div className="sidebar-section-label">YOUR WORKSPACE</div>
        <button className="new-document-button" onClick={startNewDocument}>
          <Plus size={16} />
          <span>New document</span>
          <span className="shortcut">N</span>
        </button>

        <div className="sidebar-section-label recent-label">CURRENT DOCUMENT</div>
        {documentInfo ? (
          <div className="document-row" title={documentInfo.filename}>
            <span className="document-icon"><FileText size={16} /></span>
            <span className="document-name">{documentInfo.filename}</span>
            <Check className="document-check" size={15} />
          </div>
        ) : (
          <p className="sidebar-empty">Your uploaded PDF will appear here.</p>
        )}

        <div className="sidebar-bottom">
          <div className={`api-indicator ${apiStatus}`}>
            <span className="status-dot" />
            <span>{apiStatus === "checking" ? "Connecting to API" : apiStatus === "online" ? "API connected" : "API unavailable"}</span>
          </div>
          <div className="privacy-note"><span className="privacy-lock">●</span> Your files stay in this session</div>
        </div>
      </aside>

      <div className="workspace" id="workspace">
        <header className="topbar">
          <div className="breadcrumb"><span>WORKSPACE</span><span className="crumb-divider">/</span><strong>PDF ASSISTANT</strong></div>
          <div className="topbar-right"><span className="model-indicator"><Sparkles size={14} /> GEMINI RAG</span></div>
        </header>

        <div className="workspace-grid">
          <main className="main-panel">
            {!documentInfo ? (
              <section
                className={`welcome-panel ${isDragging ? "dragging" : ""}`}
                onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
              >
                <div className="welcome-kicker"><span className="kicker-line" /> DOCUMENT INTELLIGENCE</div>
                <h1>Ask the document.<br /><em>Not the internet.</em></h1>
                <p className="welcome-copy">Bring a PDF into focus. Get grounded answers with the page to back them up.</p>

                <div className={`upload-zone ${isDragging ? "is-dragging" : ""}`}>
                  <div className="upload-icon"><ArrowDownToLine size={22} /></div>
                  <div className="upload-title">{isUploading ? "Reading your document..." : isDragging ? "Drop it right here" : "Start with a document"}</div>
                  <div className="upload-caption">Drop a PDF here, or browse your files</div>
                  <button className="browse-button" onClick={() => fileInputRef.current?.click()} disabled={isUploading}>
                    {isUploading ? <span className="spinner" /> : <Upload size={16} />}
                    {isUploading ? "Indexing PDF" : "Choose a PDF"}
                  </button>
                  <div className="file-hint">PDF ONLY <span>·</span> UP TO 20 MB</div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="application/pdf,.pdf"
                    onChange={(event) => uploadFile(event.target.files[0])}
                    hidden
                  />
                </div>
                <div className="welcome-footnote"><span className="footnote-marker">01</span> TEXT EXTRACTION <span className="footnote-arrow">→</span> CHUNKING <span className="footnote-arrow">→</span> GROUNDED ANSWERS</div>
              </section>
            ) : (
              <section className="chat-panel">
                <div className="document-summary">
                  <div className="summary-icon"><FileText size={19} /></div>
                  <div className="summary-text">
                    <div className="summary-eyebrow">DOCUMENT READY</div>
                    <h1>{documentInfo.filename}</h1>
                  </div>
                  <div className="summary-stats">
                    <span><strong>{documentInfo.page_count}</strong> pages</span>
                    <span className="stat-divider" />
                    <span><strong>{documentInfo.chunk_count}</strong> chunks</span>
                  </div>
                </div>

                <div className="conversation" aria-live="polite">
                  {messages.length === 0 ? (
                    <div className="empty-chat">
                      <div className="empty-chat-icon"><Sparkles size={18} /></div>
                      <h2>Your document is ready.</h2>
                      <p>Ask a question to find the useful parts, with page references.</p>
                      <div className="suggestion-list">
                        {["Summarize the main argument", "What are the key dates?", "What should I know first?"].map((suggestion) => (
                          <button key={suggestion} className="suggestion" onClick={() => setQuestion(suggestion)}>
                            <span>{suggestion}</span><ArrowUp size={14} />
                          </button>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="message-list">
                      {messages.map((message, index) => (
                        <article className="message-pair" key={`${message.question}-${index}`}>
                          <div className="question-row"><span className="question-avatar">Y</span><p>{message.question}</p></div>
                          <div className="answer-row">
                            <span className="answer-avatar"><Sparkles size={15} /></span>
                            <div className="answer-content">
                              {message.answer ? <p>{message.answer}</p> : <div className="thinking"><span /><span /><span /> Reading relevant passages</div>}
                              {message.sources.length > 0 && (
                                <div className="inline-citations">
                                  {message.sources.map((source, sourceIndex) => (
                                    <button key={`${source.page_numbers.join("-")}-${sourceIndex}`} onClick={() => setSelectedSource(source)}>
                                      <FileText size={12} /> {pageLabel(source.page_numbers)}
                                    </button>
                                  ))}
                                </div>
                              )}
                            </div>
                          </div>
                        </article>
                      ))}
                      <div ref={conversationEndRef} />
                    </div>
                  )}
                </div>

                <form className="question-composer" onSubmit={submitQuestion}>
                  {error && <div className="error-banner" role="alert"><CircleAlert size={15} /><span>{error}</span><button type="button" aria-label="Dismiss error" onClick={() => setError("")}><X size={15} /></button></div>}
                  <label className="sr-only" htmlFor="question-input">Ask a question about this PDF</label>
                  <textarea
                    id="question-input"
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();
                        event.currentTarget.form.requestSubmit();
                      }
                    }}
                    placeholder="Ask anything about this document..."
                    rows={2}
                    maxLength={2000}
                    disabled={isAsking}
                  />
                  <div className="composer-bottom">
                    <span>Answers are grounded in this PDF</span>
                    <div className="composer-actions"><span>{question.length}/2000</span><button type="submit" aria-label="Send question" disabled={!question.trim() || isAsking}><ArrowUp size={17} /></button></div>
                  </div>
                </form>
              </section>
            )}
            {!documentInfo && error && <div className="error-banner welcome-error" role="alert"><CircleAlert size={15} /><span>{error}</span><button type="button" aria-label="Dismiss error" onClick={() => setError("")}><X size={15} /></button></div>}
          </main>

          <aside className="sources-panel">
            <div className="sources-heading"><span>REFERENCE DESK</span><span className="sources-rule" /></div>
            {documentInfo ? (
              <>
                <div className="reference-document">
                  <div className="reference-file-icon"><FileText size={17} /></div>
                  <div><strong>{documentInfo.filename}</strong><span>{documentInfo.page_count} pages indexed</span></div>
                </div>
                <div className="sources-title-row"><h2>Source passages</h2>{latestSources.length > 0 && <span className="source-count">{latestSources.length}</span>}</div>
                {selectedSource ? (
                  <div className="selected-source">
                    <button className="source-back" onClick={() => setSelectedSource(null)}><span>←</span> ALL SOURCES</button>
                    <div className="source-page-tag"><FileText size={13} /> {pageLabel(selectedSource.page_numbers)}</div>
                    <p>{selectedSource.excerpt}</p>
                    <div className="source-relevance">MATCH {Math.round(selectedSource.relevance * 100)}%</div>
                  </div>
                ) : latestSources.length ? (
                  <div className="source-list">
                    {latestSources.map((source, index) => (
                      <button className="source-item" key={`${source.page_numbers.join("-")}-${index}`} onClick={() => setSelectedSource(source)}>
                        <span className="source-item-top"><span>{pageLabel(source.page_numbers)}</span><span>{Math.round(source.relevance * 100)}%</span></span>
                        <span className="source-excerpt">{source.excerpt}</span>
                      </button>
                    ))}
                  </div>
                ) : (
                  <p className="sources-empty">Ask a question to see the passages used to form the answer.</p>
                )}
                <div className="source-disclaimer"><Sparkles size={14} /><span>Answers use retrieved text only. No outside sources.</span></div>
              </>
            ) : (
              <div className="reference-empty">
                <div className="empty-index-mark"><BookOpenText size={22} /></div>
                <p className="reference-empty-title">Nothing indexed yet.</p>
                <p>Upload a PDF to build a searchable index and trace every answer back to its page.</p>
                <div className="reference-steps"><span>01 <i>EXTRACT</i></span><span>02 <i>INDEX</i></span><span>03 <i>ASK</i></span></div>
              </div>
            )}
            <div className="sources-footer"><span>DOCUMIND</span><span>LOCAL SESSION</span></div>
          </aside>
        </div>
      </div>
    </div>
  );
}

export default App;