import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.responses import HTMLResponse

import uvicorn
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


@app.get("/chat", response_class=HTMLResponse, include_in_schema=False)
def chat_widget() -> HTMLResponse:
    chat_html = Path(__file__).resolve().parent / "static" / "chat.html"
    return HTMLResponse(chat_html.read_text(encoding="utf-8"))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
