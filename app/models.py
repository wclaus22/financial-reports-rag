"""data models for the app"""

from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass
class RetrievedHit:
    """retrieved chunk from the document store/ vector db"""

    ticker: str
    company_name: str
    sector: str
    exchange: str
    year: int
    page_number: int
    text: str
    distance: float
    chunk_id: str
    rerank_score: float | None = None


class QueryRequest(BaseModel):
    """request model for the query endpoint"""

    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    """source model for the query response"""

    chunk_id: str
    ticker: str
    year: int
    page_number: int
    company_name: str
    sector: str
    exchange: str
    text: str
    distance: float


class QueryResponse(BaseModel):
    """response model for the query endpoint"""

    answer: str
    sources: list[Source]
