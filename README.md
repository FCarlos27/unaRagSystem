# UNASUCRE/UNASEC RAG Microservice (MVP)

Semantic search microservice over official university documents, based on
Retrieval-Augmented Generation (RAG). See [PRD.MD](./PRD.MD).

## Tech Stack

- **Backend:** FastAPI (Python)
- **Embeddings:** Nemotron-3-Embed-1B via NVIDIA API
- **LLM:** DeepSeek-R1-Distill-8B via NVIDIA API
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

1. `cp .env.example .env` and set your `NVIDIA_API_KEY`.
2. Drop official PDFs into `data/raw_pdfs/`.
3. Ingest documents: `python scripts/ingest.py`.
4. Run the API: `uvicorn app.main:app --reload` or `docker compose up --build`.

## API

- `POST /api/v1/query` — body: `{"query": "¿Cuáles son los requisitos de inscripción?"}`
- `GET /health`
