# UNASUCRE/UNASEC RAG Microservice (MVP)

Semantic search microservice over official university documents, based on
Retrieval-Augmented Generation (RAG). See [PRD.MD](./PRD.MD).

## Tech Stack

- **Backend:** FastAPI (Python)
- **Embeddings:** nomic-embed-text (local via Ollama)
- **LLM:** Llama-3.2-3B (local via Ollama)
- **Vector DB:** ChromaDB (local)
- **Orchestration:** LangChain
- **PDF parsing:** PyPDF2
- **Chunking:** RecursiveCharacterTextSplitter (LangChain)
- **Infrastructure:** Docker

## Folder Structure

```
.
├── app/
│   ├── api/routes/       # FastAPI route handlers (POST /api/v1/query)
│   ├── core/             # Configuration (pydantic-settings)
│   ├── schemas/          # Pydantic request/response models
│   ├── services/         # RAG pipeline (loader, splitter, embeddings, store, llm)
│   └── utils/            # Logging
├── data/
│   ├── raw_pdfs/         # Source PDFs for ingestion
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
3. Drop official PDFs into `data/raw_pdfs/`.
4. Ingest documents: `python scripts/ingest.py`.
5. Run the API: `uvicorn app.main:app --reload` or `docker compose up --build`.

## API

- `POST /api/v1/query` — body: `{"query": "¿Cuáles son los requisitos de inscripción?"}`
  - Omit `session_id` to start a new conversation (one is returned in the response).
  - Reuse the returned `session_id` to keep conversation context for follow-up questions.
- `GET /health`

Conversation history is stored in memory (resets on restart).
