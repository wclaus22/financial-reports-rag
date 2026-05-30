"""data models"""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel


class CompanyMetadata(BaseModel):
    """metadata about a company"""

    name: str
    ticker: str
    sector: str
    exchange: str


@dataclass
class PageSegment:
    """A semantically-coherent slice of a page: prose paragraph(s) or one table."""

    kind: Literal["prose", "table"]
    text: str
    section_title: str | None = None


@dataclass
class PageData:
    ticker: str
    company_name: str
    sector: str
    exchange: str
    year: int
    page_number: int  # page numbers are 1-indexed
    text: str  # full page text — debug/introspection, not embedded directly
    segments: list[PageSegment] = field(default_factory=list)


@dataclass
class Chunk:
    ticker: str
    company_name: str
    sector: str
    exchange: str
    year: int
    page_number: int  # page numbers are 1-indexed
    chunk_id: str
    text: str
