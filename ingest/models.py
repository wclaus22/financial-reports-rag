"""data models"""

from dataclasses import dataclass

from pydantic import BaseModel


class CompanyMetadata(BaseModel):
    """metadata about a company"""

    name: str
    ticker: str
    sector: str
    exchange: str


@dataclass
class PageData:
    ticker: str
    company_name: str
    sector: str
    exchange: str
    year: int
    page_number: int  # page numbers are 1-indexed
    text: str


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
