from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language question from the student")


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
