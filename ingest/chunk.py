"""split page-level text into retrieval sized chunks while preserving metadata"""

import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingest.models import Chunk, PageData


def chunk_pages(
    pages: list[PageData], chunk_size: int = 1000, chunk_overlap: int = 100
):
    """Split page-level text into retrieval sized chunks while preserving metadata."""
    chunks: list[Chunk] = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for page in tqdm.tqdm(pages, desc="Chunking pages", total=len(pages)):
        page_chunks = text_splitter.create_documents([page.text])
        for i, chunk in enumerate(page_chunks):
            chunk_id = f"{page.ticker}_{page.year}_page{page.page_number}_chunk{i + 1}"
            chunks.append(
                Chunk(
                    ticker=page.ticker,
                    company_name=page.company_name,
                    sector=page.sector,
                    exchange=page.exchange,
                    year=page.year,
                    page_number=page.page_number,
                    chunk_id=chunk_id,
                    text=chunk.page_content,
                )
            )

    return chunks
