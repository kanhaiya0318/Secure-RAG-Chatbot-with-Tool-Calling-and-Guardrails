# FastAPI RAG ChatBot

A document-aware chatbot backend built with **FastAPI**, the **Gemini API** (called directly over `httpx`) and **ChromaDB**. Upload PDFs and ask questions: the model decides when to search your documents or the web, and answers with the source and page. It includes streaming replies, conversation memory, layered guardrails, LangSmith tracing, and a **Streamlit** chat UI.

## What stands out

1. **Layered guardrails.** Five layers go beyond "it answers questions": prompt-injection handling (retrieved text is wrapped as untrusted data), Gemini safety-block handling that keeps the chat history valid, rate and cost limits, hardened PDF uploads, and output checks (secret redaction, system-prompt leak detection, source line on document answers). *See `app/guardrails.py`, `app/limits.py`, `app/rag/validation.py`.*
2. **Failure-mode engineering.** History trimming never separates a tool call from its response, a blocked turn is removed so it cannot poison later messages, and every tool call gets a response even when the per-turn cap is hit. *See `app/services/memory_service.py`, `app/services/chat_service.py`.*
3. **No framework glue.** A direct Gemini REST client, a hand-written tool-calling loop, and a self-contained RAG package (loader, splitter, embeddings, vector store, retriever, pipeline) with a one-way dependency direction. *See `app/rag/`.*
4. **Observability and cost control.** Every chat turn is traced in LangSmith, with each model call and tool call as its own span. Rate limits, upload caps and a tool-call cap bound spend, and an answer cache skips the model call for repeated document questions. *See `app/cache.py`.*
5. **Engineering hygiene.** Typed settings loaded from `.env`, typed request and response schemas, content-hash deduplication of uploads, secrets kept out of the repository, and documented known limitations.


## Features

- **RAG over PDFs:** PyMuPDF loader, text cleaning, recursive splitter, Gemini embeddings, Chroma vector store (cosine distance)
- **Two tools:** `SearchDocs` searches your uploaded documents and `WebSearch` searches the web through Tavily. The model chooses, and may use both in one turn
- **Streaming replies** from `streamGenerateContent`, shown live in the UI
- **Duplicate-safe uploads:** documents are identified by a hash of their content, so re-uploading the same file is skipped
- **Session-based chat** with memory: a sliding window plus a rolling summary keeps the prompt small on long chats
- **Answer cache** for repeated document questions (see [Answer cache](#answer-cache))
- **Streamlit UI:** multiple chats, PDF upload, document list and delete
- **Framework-free RAG package** (`app/rag/`): no imports from routers or dependencies

## Architecture

```
Streamlit UI ──HTTP──▶ FastAPI
                        ├── /chat/query ──▶ rate limits ──▶ answer cache ──hit──▶ reply
                        │                                        │ miss
                        │                                        ▼
                        │                  chat_service ──▶ Gemini (streaming)
                        │                        │  functionCall
                        │                        ├── SearchDocs ──▶ RAGPipeline.retrieve ──▶ Chroma
                        │                        └── WebSearch  ──▶ Tavily
                        ├── /rag/*      ──▶ validation ──▶ RAGPipeline.ingest ──▶ Chroma
                        └── /sessions/* ──▶ in-memory session store
```

**Ingest** (`POST /rag/upload`): rate limit → size, type, page and encryption checks → hash file and skip if already indexed → extract text per page → clean → split into chunks → embed with Gemini → store in Chroma.

**Query** (`POST /chat/query`): the model receives the question and calls `SearchDocs` and/or `WebSearch` when needed. Tool results come back wrapped as untrusted data. The final answer streams through an output filter that redacts secrets and detects system-prompt leaks.

## Guardrails

| Layer | What it does | Where |
|---|---|---|
| **1. Untrusted content** | Retrieved and web text is wrapped in `<untrusted_data>` tags with a notice. The system prompt tells the model to ignore instructions found in tool results. Obvious injection phrases are logged | `guardrails.py`, `tools.py`, `rag/prompts.py` |
| **2. Model safety** | Gemini `safetySettings` on four categories. Blocked prompts, blocked responses and malformed tool calls give the user a clear message, and the blocked turn is removed from history | `llm_service.py`, `guardrails.py`, `chat_service.py` |
| **3. Abuse and cost limits** | Rate limits per IP, per session and for uploads (429). Session cap with idle expiry (503). Cap on tool calls per turn and on search query length | `limits.py`, `chat_service.py`, `tools.py` |
| **4. Upload hardening** | `.pdf` name, streamed size limit, `%PDF` header and parse check, encrypted or corrupt files rejected, page limit, document cap, filename cleaning | `rag/validation.py`, `routers/rag.py` |
| **5. Output checks** | Secrets in answers are replaced with `[redacted]`. A hidden marker in the system prompt detects verbatim prompt leaks. A "Sources consulted" line is added when the model names no source | `guardrails.py`, `chat_service.py` |

## Answer cache

Repeated document questions can skip the model call entirely. An answer is **stored** only when all of these hold:

- the question was the **first message of its session** (later messages depend on earlier chat context)
- the turn used **only `SearchDocs`** (web answers go stale, and tool-free answers can depend on the current date)
- no tool returned an error, and the answer was not blocked or cut off

An answer is **used** only for the first message of a session, when the same question (ignoring case, spacing and trailing `?.!`) is already stored for the same model. The cache is cleared when a document is uploaded or deleted, and entries expire after `ANSWER_CACHE_TTL_SECONDS` or are evicted when it grows past `ANSWER_CACHE_MAX_ENTRIES`. It lives in memory, so a restart empties it.

## Project structure

```
.
├── app/
│   ├── main.py                  # app, lifespan (HTTP client + RAG pipeline built once), timing middleware
│   ├── config.py                # typed settings loaded from .env
│   ├── dependancies.py          # get_db, get_llm_service, build/get_rag_pipeline
│   ├── guardrails.py            # untrusted-content wrapping, block detection, output filter, sources line
│   ├── limits.py                # rate limiters, session cap
│   ├── cache.py                 # answer cache
│   ├── rag/
│   │   ├── schema.py            # Document(page_content, metadata)
│   │   ├── preprocessing.py     # text cleaning
│   │   ├── prompts.py           # system prompt and summary prompt
│   │   ├── validation.py        # PDF checks
│   │   ├── pipeline.py          # RAGPipeline: ingest, retrieve, list, delete
│   │   ├── loaders/             # PyMuPDFLoader
│   │   ├── splitters/           # RecursiveCharacterTextSplitter
│   │   ├── embeddings/          # GeminiEmbeddings
│   │   ├── vectorstores/        # ChromaVectorStore
│   │   └── retrievers/          # VectorStoreRetriever
│   ├── services/
│   │   ├── llm_service.py       # Gemini REST client (stream + summarize)
│   │   ├── chat_service.py      # streaming reply loop, tool execution, cache use
│   │   ├── memory_service.py    # sliding window + rolling summary
│   │   └── tools.py             # SearchDocs, WebSearch
│   ├── routers/                 # chat.py, sessions.py, rag.py
│   ├── schemas/                 # chat.py, rag.py
│   └── utils/timing.py
├── tests/test_guardrails.py     # API-level guardrail tests
├── streamlit_api.py             # chat UI
├── requirements.txt
├── pyproject.toml
└── .env                         # not committed
```

Dependency direction: `routers → dependancies → services → rag`. Services never import from `dependancies`, which avoids circular imports.

## Setup

### 1. Install

Requires **Python 3.12+**.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure `.env`

Create a `.env` file in the project root:

```env
# required
GEMINI_API_KEY=your_gemini_key
GEMINI_EMBED_MODEL=gemini-embedding-001
TAVILY_API_KEY=your_tavily_key

# optional: LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=fastapi-rag-chatbot
```

Everything else has a default, listed below.

> **Changing `OUTPUT_DIM` or the embedding model?** Delete `data/chroma` and re-upload your PDFs. A Chroma collection only accepts vectors of the size it was created with.

### 3. Run

Use two terminals, both from the **project root**. The Chroma index is stored in `data/chroma`, relative to where you launch the server.

```bash
# Terminal 1: backend
uvicorn app.main:app --reload

# Terminal 2: UI
streamlit run streamlit_api.py
```

- API docs: http://127.0.0.1:8000/docs
- UI: http://localhost:8501

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | required | Gemini API key |
| `GEMINI_EMBED_MODEL` | required | Embedding model id |
| `TAVILY_API_KEY` | required | Tavily key for `WebSearch` |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite` | Model for chat and summaries |
| `RETRY_LLM_CALL` | `5` | Max model/tool turns per user message |
| `TRIGGER` | `8` | Summarize when history has more than this many messages |
| `KEEP_LAST` | `4` | Approximate number of recent messages kept after summarizing |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `700` / `100` | Chunk length and overlap, in characters |
| `OUTPUT_DIM` | `1024` | Embedding vector size |
| `BATCH_SIZE` | `10` | Chunks per embedding request |
| `CHAT_PER_MIN_IP` | `30` | Chat requests per minute per IP |
| `CHAT_PER_MIN_SESSION` | `10` | Chat requests per minute per session |
| `UPLOADS_PER_10MIN` | `5` | Uploads per 10 minutes per IP |
| `MAX_SESSIONS` | `200` | Sessions held in memory |
| `SESSION_TTL_SECONDS` | `3600` | Idle time before a session can be dropped when the cap is reached |
| `MAX_TOOL_CALLS_PER_TURN` | `3` | Tool calls the model may make in one turn |
| `MAX_QUERY_CHARS` | `300` | Maximum length of a search query |
| `MAX_UPLOAD_MB` | `20` | Maximum PDF size |
| `MAX_PDF_PAGES` | `100` | Maximum pages per PDF |
| `MAX_DOCUMENTS` | `20` | Maximum indexed documents |
| `MAX_FILENAME_CHARS` | `80` | Maximum stored filename length |
| `ANSWER_CACHE_ENABLED` | `true` | Turn the answer cache on or off |
| `ANSWER_CACHE_MAX_ENTRIES` | `200` | Maximum cached answers |
| `ANSWER_CACHE_TTL_SECONDS` | `3600` | Cached answer lifetime |

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/chat/query` | Send a message, receive the reply as a text stream |
| POST | `/rag/upload` | Upload and index a PDF |
| GET | `/rag/documents` | List indexed documents |
| DELETE | `/rag/documents/{doc_id}` | Delete a document and its chunks |
| GET | `/sessions/history/{session_id}` | Get the summary and history of a session |
| GET | `/sessions/get_session_ids` | List all session ids |
| POST | `/sessions/{session_id}/end` | Delete a session |

**`POST /chat/query`**

```json
// request (message: 1-200 characters, session_id: 3+ characters)
{ "session_id": "abc-123", "message": "Which university did the candidate attend?" }
```

The response is `text/plain`, streamed in chunks. Errors: `422` invalid input, `429` rate limited (with `Retry-After`), `503` server at session capacity.

**`POST /rag/upload`**: multipart form with a `file` field. PDF only.

```json
// response
{ "doc_id": "cde5bfbf2e85f48f", "source": "report.pdf", "total_pages": 2, "chunks": 8, "skipped": false }
```

`skipped` is `true` when the same file content was already indexed. Errors: `400` not a PDF, `409` document limit reached, `413` file too large, `422` empty, corrupt, encrypted, too many pages, or no extractable text, `429` too many uploads.

Example with curl:

```bash
curl -F "file=@report.pdf" http://127.0.0.1:8000/rag/upload
curl -N -X POST http://127.0.0.1:8000/chat/query \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-1","message":"Summarize the document"}'
```

## How memory works

Each session is stored as:

```python
db[session_id] = {"summary": "", "history": [...], "last_active": ...}   # history is in Gemini message format
```

On every message:

1. The user message is appended to `history`.
2. If `len(history) > TRIGGER`, older messages are summarized and removed; roughly the last `KEEP_LAST` messages stay (the sliding window).
3. The old summary and the dropped messages go to Gemini to produce a **new rolling summary**.
4. The summary is passed to the model through the system instruction on later calls.

The cut point must land on a plain user message (one with text, not a tool response), so a `functionCall` is never separated from its `functionResponse`.

## Tool calling flow

1. The model replies with one or more `functionCall` parts (`SearchDocs`, `WebSearch`).
2. The server runs the tools (at most `MAX_TOOL_CALLS_PER_TURN`; extra calls get an error response) and appends a `functionResponse` turn to history.
3. The model is called again, up to `RETRY_LLM_CALL` times, until it returns a final text answer.

Tools always return a dict, including `{"error": "..."}` on failure, so the model can explain problems to the user.

## Observability

With `LANGSMITH_TRACING=true`, each user message becomes one `chat_turn` trace in LangSmith:

```
chat_turn
 ├─ gemini_stream   (llm: decides to call a tool)
 ├─ SearchDocs      (tool: query in, chunks out)
 └─ gemini_stream   (llm: final answer)
```

Chat turns carry the session id as metadata, so a conversation appears as one thread. A cache hit shows up as a `chat_turn` with no model span. Traces include prompts and retrieved text, so check LangSmith's options for hiding inputs and outputs before tracing private documents. The server also logs step timings, and every response carries an `X-Process-Time-ms` header.

## Testing

`tests/test_guardrails.py` runs against a live server:

```bash
python tests/test_guardrails.py make        # create PDFs for manual chat tests
python tests/test_guardrails.py l4          # upload hardening (set UPLOADS_PER_10MIN=100 first)
python tests/test_guardrails.py l3          # rate limits (restart the server first)
python tests/test_guardrails.py cap 3       # session cap  (set MAX_SESSIONS=3)
python tests/test_guardrails.py doccap 2    # document cap (set MAX_DOCUMENTS=2)
```

## Limitations

- Sessions, history, rate-limit counters and the answer cache live **in memory**: restarting the server clears them. The Chroma index persists on disk.
- Single-process design: not intended for multiple workers.
- The Streamlit chat list lives in `st.session_state`, so refreshing the browser tab clears it.
- PDFs only, text-based. Scanned PDFs without a text layer are rejected.
- Text cleaning currently keeps English text only.
- Messages are limited to 200 characters.
- Memory trigger uses message count, not tokens.
- The answer cache matches exact questions only, and only for the first message of a session.
- Personal data such as emails and phone numbers is not redacted from answers, on purpose, since document questions often need it. Secret redaction covers common key formats only.
- The system-prompt leak check catches verbatim leaks, not paraphrases.
- The upload size check cannot stop the request body itself; a hard cap needs a reverse proxy.
- There is no relevance threshold or claim verification for document answers yet.