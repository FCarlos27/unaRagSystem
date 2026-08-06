from fastapi import APIRouter, HTTPException

from app.schemas.query import QueryRequest, QueryResponse
from app.services.rag_engine import get_rag_engine
from app.utils.logging import get_logger

logger = get_logger("query_route")

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest) -> QueryResponse:
    try:
        engine = get_rag_engine()
        answer, sources, confidence = engine.answer(request.query)
        return QueryResponse(answer=answer, sources=sources, confidence=confidence)
    except Exception as exc:
        logger.exception("Failed to answer query")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
