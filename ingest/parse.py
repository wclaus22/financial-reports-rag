"""parse PDFs into page-level text with metadata derived from file path"""

import json
from pathlib import Path

import pypdf
import tqdm

from ingest.models import CompanyMetadata, PageData


def parse_metadata(file_path: str) -> dict[str, CompanyMetadata]:
    """Parse metadata from a file path."""
    # extract company name and year from file path
    # e.g. data/roche_2022.pdf -> company_name=roche, year=2022

    with open(file_path, "r", encoding="utf-8") as f:
        company_metadata = json.load(f)

    return {
        folder: CompanyMetadata(**data) for folder, data in company_metadata.items()
    }


def parse_pdf(metadata_path: str) -> list[PageData]:
    """Parse every PDF under data/raw/ and return a flat list of PageData."""
    metadata = parse_metadata(metadata_path)

    pages: list[PageData] = []
    pdf_paths = sorted(Path("data/raw").glob("*/*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError("No PDF files found in data/raw/")

    for pdf_path in pdf_paths:
        company_folder = pdf_path.parent.name
        year = int(pdf_path.stem.split("_")[1])
        company_metadata = metadata[company_folder]
        reader = pypdf.PdfReader(pdf_path)

        for i, page in tqdm.tqdm(
            enumerate(reader.pages),
            total=len(reader.pages),
            desc=f"Processing {pdf_path.name}",
        ):
            try:
                text = page.extract_text() or ""
            except Exception as e:
                print(f"  WARN: failed to extract page {i + 1} of {pdf_path.name}: {e}")
                continue
            text = text.strip()
            if not text:
                continue

            pages.append(
                PageData(
                    ticker=company_metadata.ticker,
                    company_name=company_metadata.name,
                    sector=company_metadata.sector,
                    exchange=company_metadata.exchange,
                    year=year,
                    page_number=i + 1,
                    text=text,
                )
            )

    return pages


if __name__ == "__main__":
    parse_pdf("data/company_metadata.json")
    breakpoint()
