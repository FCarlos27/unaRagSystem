# UNASUCRE/UNASEC RAG Microservice (MVP)

Semantic search microservice over official university documents, based on
Retrieval-Augmented Generation (RAG). See [PRD.MD](./PRD.MD).

## Tech Stack

- **Backend:** FastAPI (Python)
- **Embeddings:** nomic-embed-text (local via Ollama)
- **LLM:** Llama-3.2-3B (local via Ollama)
- **Vector DB:** ChromaDB (local)
- **Orchestration:** LangChain
- **Document parsing:** python-docx (.docx) + JSON ingestion
- **Chunking:** RecursiveCharacterTextSplitter (LangChain)
- **Infrastructure:** Docker

## Architecture: Waterfall Router Pattern

The RAG microservice implements a **three-tier execution flow** that short-circuits processing for trivial or rule-based queries, ensuring sub-5ms responses for common interactions while retaining full semantic power for complex questions.

```
User Query
    │
    ▼
┌──────────────────────┐   Matched?  ──► Return canned response (<1ms)
│ Tier 1: Heuristics   │              │  e.g. greetings, thanks, farewells
└──────────────────────┘              │
    │ No                            │
    ▼                              │
┌──────────────────────┐  Resolved? ──► Return formatted rule data (50-150ms)
│ Tier 2: Deterministic│             │  e.g. bank accounts, coordinator emails
└──────────────────────┘             │
    │ No                           │
    ▼                             │
┌──────────────────────┐           │
│ Tier 3: Semantic RAG │◄── Always run vector search + LLM generation (~500-1500ms)
└──────────────────────┘
    │
    ▼
Answer returned with sources & confidence
```

### Latency Expectations

| Tier | Description | Typical Latency | Implementation |
|------|-------------|-----------------|----------------|
| Heuristics | Greetings, thanks, farewells | <1ms | `app/services/rules.py` (`match_heuristic()`) |
| Deterministic | Bank accounts, coordinator emails | 50-150ms | `app/services/rules.py` (`resolve_deterministic()`) |
| Semantic RAG | Vector retrieval + Llama 3.2 | ~500-1500ms | `app/services/rag_engine.py` (`RagEngine.answer()`) |

### Implementation Notes

- **Heuristics** (`match_heuristic`): Uses anchored regex patterns to detect conversational phrases. Bypasses vector DB entirely.
- **Deterministic** (`resolve_deterministic`): Uses keyword patterns plus metadata inspection (JSON documents) to return exact banking or contact data. Requires vector search results but skips LLM generation.
- **Semantic RAG**: Full retrieval-augmented generation pipeline. Includes query reformulation for follow-up questions (skipped for heuristic matches or empty history).

All tiers update session history consistently for multi-turn conversations.

## Folder Structure

```
.
├── app/
│   ├── core/             # Configuration (pydantic-settings)
│   ├── schemas/          # Pydantic request/response models
│   ├── services/         # RAG pipeline (loader, splitter, embeddings, store, llm, rules, rag_engine)
│   └── utils/            # Logging
├── data/
│   ├── raw_docx/         # Source .docx instructions for ingestion
│   ├── json_docs/        # Structured JSON data (banks, directories, contacts)
│   └── chroma_db/        # Persistent ChromaDB storage
├── scripts/              # Ingestion script
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Setup

1. Ensure Ollama is running locally and the models are available:
   `ollama pull nomic-embed-text` and `ollama pull llama3.2:3b`
2. `cp .env.example .env`.
3. Drop official `.docx` instructions into `data/raw_docx/` and JSON files into `data/json_docs/`.
4. Ingest documents: `python scripts/ingest.py`.
5. Run the API: `uvicorn app.main:app --reload` or `docker compose up --build`.

## API

- `POST /api/v1/query` — body: `{"query": "¿Cuáles son los requisitos de inscripción?"}`
  - Omit `session_id` to start a new conversation (one is returned in the response).
  - Reuse the returned `session_id` to keep conversation context for follow-up questions.
- `GET /health`

Conversation history is stored in memory (resets on restart).
