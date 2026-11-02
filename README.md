# FastAPI RAG ChatBot

A document-aware chatbot backend built with **FastAPI**, the **Gemini API** (called directly over `httpx`) and **ChromaDB**. Upload PDFs, then ask questions: the model decides when to search your documents and answers with the source and page. Includes streaming replies, conversation memory (sliding window + rolling summary) and a **Streamlit** chat UI.

## What stands out

1. **Layered guardrails.** Five layers go beyond "it answers questions": prompt-injection handling (retrieved text is wrapped as untrusted data), Gemini safety-block handling that keeps the chat history valid, rate and cost limits, hardened PDF uploads, and output checks (secret redaction, system-prompt leak detection, source line on document answers). *See `app/guardrails.py`, `app/limits.py`, `app/rag/validation.py`.*
2. **Failure-mode engineering.** History trimming never separates a tool call from its response, a blocked turn is removed so it cannot poison later messages, and every tool call gets a response even when the per-turn cap is hit. *See `app/services/memory_service.py`, `app/services/chat_service.py`.*
3. **No framework glue.** A direct Gemini REST client, a hand-written tool-calling loop, and a self-contained RAG package (loader, splitter, embeddings, vector store, retriever, pipeline) with a one-way dependency direction. *See `app/rag/`.*
4. **Observability.** Every chat turn is traced in LangSmith, with each model call and tool call as its own span, plus request and step timing logs.
5. **Engineering hygiene.** Typed settings loaded from `.env`, typed request and response schemas, content-hash deduplication of uploads, secrets kept out of the repository, and documented known limitations.

## Pending (future work)

- [ ] **Evaluation:** build a question set from sample documents and report retrieval recall@k and answer faithfulness, then compare chunk sizes and `k` using those numbers instead of guesses.
- [ ] **Automated tests and CI:** turn `test_guardrails.py` into a pytest suite (guardrails, splitter, memory logic) and run it on every push.
- [ ] **Deployment:** Dockerfile, a hosted demo link, and a short demo video.
- [ ] **Persistent sessions:** move sessions and history from memory to a database or Redis, which also lets rate limits work across several workers.
- [ ] **Caching:**
  - [ ] Cache query embeddings, so repeated questions skip the embedding API call
  - [ ] Cache `WebSearch` results with a short expiry
  - [ ] Cache answers to identical document questions, keyed by the document set and the normalized question, and clear them when a document is uploaded or deleted

## Features

- **RAG over PDFs**: PyMuPDF loader, text cleaning, recursive splitter, Gemini embeddings, Chroma vector store
- **Tool calling**: the model calls a `SearchDocs` tool to retrieve relevant chunks, then writes the final answer
- **Streaming replies** from `streamGenerateContent`, shown live in the UI
- **Duplicate-safe uploads**: documents are identified by a hash of their content, so re-uploading the same file is skipped
- **Session-based chat** with memory: a sliding window plus a rolling summary keeps the prompt small on long chats
- **Streamlit UI**: multiple chats, PDF upload, document list and delete
- **Framework-free RAG package** (`app/rag/`): no imports from routers or dependencies

## Architecture

```
Streamlit UI ──HTTP──▶ FastAPI
                        ├── /chat/query  ──▶ chat_service ──▶ Gemini (streaming)
                        │                        │  functionCall: SearchDocs
                        │                        ▼
                        │                   RAGPipeline.retrieve ──▶ Chroma
                        ├── /rag/*       ──▶ RAGPipeline.ingest   ──▶ Chroma
                        └── /sessions/*  ──▶ in-memory session store
```

**Ingest** (`POST /rag/upload`): hash file → skip if already indexed → extract text per page → clean → split into chunks → embed with Gemini → store in Chroma.

**Query** (`POST /chat/query`): the model receives the question, calls `SearchDocs` when the answer may be in your documents, receives the top chunks (text, source, page), and streams a final answer.

## Project structure

```
.
├── app/
│   ├── main.py                  # app, lifespan (HTTP client + RAG pipeline built once)
│   ├── config.py                # settings loaded from .env
│   ├── dependancies.py          # get_db, get_llm_service, build/get_rag_pipeline
│   ├── rag/
│   │   ├── schema.py            # Document(page_content, metadata)
│   │   ├── preprocessing.py     # text cleaning
│   │   ├── prompts.py           # system prompt and summary prompt
│   │   ├── pipeline.py          # RAGPipeline: ingest, retrieve, list, delete
│   │   ├── loaders/             # PyMuPDFLoader
│   │   ├── splitters/           # RecursiveCharacterTextSplitter
│   │   ├── embeddings/          # GeminiEmbeddings
│   │   ├── vectorstores/        # ChromaVectorStore
│   │   └── retrievers/          # VectorStoreRetriever
│   ├── services/
│   │   ├── llm_service.py       # Gemini REST client (stream + summarize)
│   │   ├── chat_service.py      # streaming reply loop + tool execution
│   │   ├── memory_service.py    # sliding window + rolling summary
│   │   └── tools.py             # SearchDocs (enabled), WebSearch (disabled)
│   ├── routers/                 # chat.py, sessions.py, rag.py
│   ├── schemas/                 # chat.py, rag.py
│   └── utils/timing.py
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

# optional (defaults shown)
GEMINI_MODEL=gemini-3.1-flash-lite
RETRY_LLM_CALL=5
TRIGGER=8
KEEP_LAST=4
CHUNK_SIZE=700
CHUNK_OVERLAP=100
OUTPUT_DIM=1024
BATCH_SIZE=1
SEARCH_DEPTH=ultra-fast
SEARCH_MAX_RESULTS=5
```

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Gemini API key |
| `GEMINI_EMBED_MODEL` | Embedding model id |
| `TAVILY_API_KEY` | Tavily key (required by settings; web search is currently disabled) |
| `GEMINI_MODEL` | Model used for chat and summaries |
| `RETRY_LLM_CALL` | Max model/tool turns per user message |
| `TRIGGER` | Summarize when history has more than this many messages |
| `KEEP_LAST` | Approximate number of recent messages kept after summarizing |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunk length and overlap, in characters |
| `OUTPUT_DIM` | Embedding vector size |
| `BATCH_SIZE` | Chunks per embedding request |
| `SEARCH_DEPTH` / `SEARCH_MAX_RESULTS` | Tavily web search options |

> **Changing `OUTPUT_DIM` or the embedding model?** Delete `data/chroma` and re-upload your PDFs. A Chroma collection only accepts vectors of the size it was created with.

### 3. Run

Use two terminals, both from the **project root** (the Chroma index is stored in `data/chroma`, relative to where you launch the server).

```bash
# Terminal 1: backend
uvicorn app.main:app --reload

# Terminal 2: UI
streamlit run streamlit_api.py
```

- API docs: http://127.0.0.1:8000/docs
- UI: http://localhost:8501

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

The response is `text/plain`, streamed in chunks.

**`POST /rag/upload`**: multipart form with a `file` field. PDF only, up to 20 MB.

```json
// response
{ "doc_id": "cde5bfbf2e85f48f", "source": "report.pdf", "total_pages": 2, "chunks": 8, "skipped": false }
```

`skipped` is `true` when the same file content was already indexed. Errors: `400` not a PDF, `413` file too large, `422` no extractable text (for example a scanned PDF).

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
db[session_id] = {"summary": "", "history": [...]}   # history is in Gemini message format
```

On every message:

1. The user message is appended to `history`.
2. If `len(history) > TRIGGER`, older messages are summarized and removed; roughly the last `KEEP_LAST` messages stay (the sliding window).
3. The old summary and the dropped messages go to Gemini to produce a **new rolling summary**.
4. The summary is passed to the model through the system instruction on later calls.

The cut point must land on a plain user message (one with text, not a tool response), so a `functionCall` is never separated from its `functionResponse`.

## Tool calling flow

1. The model replies with a `functionCall` (for example `SearchDocs`).
2. The server runs the tool and appends a `functionResponse` turn to history.
3. The model is called again, up to `RETRY_LLM_CALL` times, until it returns a final text answer.

Tools always return a dict, including `{"error": "..."}` on failure, so the model can explain problems to the user.

## Limitations

- Sessions and history live **in memory**: restarting the server clears them. The Chroma index persists on disk.
- The Streamlit chat list lives in `st.session_state`, so refreshing the browser tab clears it.
- PDFs only, text-based. Scanned PDFs without a text layer are rejected.
- Text cleaning currently keeps English text only.
- Messages are limited to 200 characters.
- Memory trigger uses message count, not tokens.
- Single-process design: not intended for multiple workers.

## Roadmap

- [ ] Enable the web search tool alongside `SearchDocs`
- [ ] Token-based summarization trigger using `usageMetadata.promptTokenCount`
- [ ] Persistent storage (database) for sessions and history
- [ ] Support for more file types (DOCX, TXT) and scanned PDFs (OCR)
- [ ] Unit tests for the splitter, loader, vector store and memory logic
- [ ] Docker setup