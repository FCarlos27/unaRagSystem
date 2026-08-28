import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

import uvicorn

from app.schemas.query import QueryRequest, QueryResponse
from app.services.rag_engine import get_rag_engine
from app.utils.logging import setup_logging, get_logger

logger = setup_logging()
query_logger = get_logger("query_route")

app = FastAPI(
    title="UNASUCRE/UNASEC RAG Microservice",
    description="Semantic search over official university documents.",
    version="0.1.0",
)


@app.post("/api/v1/query", response_model=QueryResponse)
def handle_query(request: QueryRequest) -> QueryResponse:
    try:
        engine = get_rag_engine()
        answer, sources, confidence, session_id = engine.answer(
            request.query, request.session_id
        )
        return QueryResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            session_id=session_id,
        )
    except Exception as exc:
        query_logger.exception("Failed to answer query")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health", tags=["health"])
def health_check() -> dict:
    return {"status": "ok"}


@app.get("/chat", response_class=HTMLResponse, include_in_schema=False)
def chat_widget() -> HTMLResponse:
    chat_html = Path(__file__).resolve().parent / "static" / "chat.html"
    return HTMLResponse(chat_html.read_text(encoding="utf-8"))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
