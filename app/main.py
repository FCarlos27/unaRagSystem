from fastapi import FastAPI

from app.api.routes.query import router as query_router
from app.utils.logging import setup_logging

logger = setup_logging()

app = FastAPI(
    title="UNASUCRE/UNASEC RAG Microservice",
    description="Semantic search over official university documents.",
    version="0.1.0",
)

app.include_router(query_router)


@app.get("/health", tags=["health"])
def health_check() -> dict:
    return {"status": "ok"}
