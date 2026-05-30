"""split page-level segments into retrieval-sized chunks while preserving metadata.

Two key behaviours:

  - Every chunk is prefixed with an identifying header
    (`"{Company} ({Ticker}) — Annual Report {Year} — p.{N}[ — {heading}]"`)
    so the embedding sees who/when/where for every chunk, not just the ones
    that happened to grab the page title.
  - Table segments are emitted as a SINGLE chunk regardless of size — keeping
    column headers and row values together is the whole point of segmenting
    tables out at parse time. Prose segments still go through the recursive
    character splitter.
"""

import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingest.models import Chunk, PageData


def _page_heading(text: str) -> str:
    """First non-trivial line of a page, stripped of a leading bare page number.

    Returns '' when nothing usable shows up in the first few lines.
    """
    for line in text.split("\n", 5):
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        if parts[0].isdigit() and len(parts) > 1:
            line = parts[1].strip()
        if not line or len(line) > 100:
            return ""
        if line.endswith(".") and len(line) > 40:
            return ""
        return line
    return ""


def _make_chunk(page: PageData, idx: int, prefix: str, body: str) -> Chunk:
    return Chunk(
        ticker=page.ticker,
        company_name=page.company_name,
        sector=page.sector,
        exchange=page.exchange,
        year=page.year,
        page_number=page.page_number,
        chunk_id=f"{page.ticker}_{page.year}_page{page.page_number}_chunk{idx}",
        text=f"{prefix}\n\n{body}",
    )


def chunk_pages(
    pages: list[PageData], chunk_size: int = 1000, chunk_overlap: int = 100
):
    """Split page segments into retrieval-sized chunks while preserving metadata."""
    chunks: list[Chunk] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for page in tqdm.tqdm(pages, desc="Chunking pages", total=len(pages)):
        heading = _page_heading(page.text)
        base_prefix = (
            f"{page.company_name} ({page.ticker}) — "
            f"Annual Report {page.year} — p.{page.page_number}"
        )
        if heading:
            base_prefix += f" — {heading}"

        idx = 0
        for seg in page.segments:
            if seg.kind == "prose":
                pieces = [
                    d.page_content for d in splitter.create_documents([seg.text])
                ]
                for piece in pieces:
                    idx += 1
                    chunks.append(_make_chunk(page, idx, base_prefix, piece))
            else:  # table — emit as a single chunk
                idx += 1
                table_prefix = base_prefix
                if seg.section_title and seg.section_title not in base_prefix:
                    table_prefix = f"{base_prefix} — Table: {seg.section_title}"
                else:
                    table_prefix = f"{base_prefix} — Table"
                chunks.append(_make_chunk(page, idx, table_prefix, seg.text))

    return chunks
