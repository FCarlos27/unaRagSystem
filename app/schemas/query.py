from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language question from the student")
    session_id: str | None = Field(
        default=None,
        description="Session id for follow-up conversation context. Omit for a new conversation.",
    )


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
    session_id: str
