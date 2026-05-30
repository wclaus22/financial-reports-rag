"""
parse PDFs into page-level segments (prose + tables) with metadata derived
from file path.

NOTE on page_number: this is the PDF *leaf* index (1-based), NOT the printed
footer page. They differ by the size of the front matter. Citations should be
labelled as "PDF p. X" until/unless we add printed-page extraction.

Switched from pypdf to pdfplumber so tables are reconstructed WITH structure
instead of linearized into bare rows of numbers.

Tables are emitted as *standalone* PageSegments so the chunker can keep each
table together as a single retrieval unit (instead of splitting column headers
away from row values). Two sources of tables:

  1) pdfplumber's `extract_tables()` (ruled-line tables) — serialized to GFM
     markdown.
  2) a regex-based heuristic over `extract_text()` lines that catches
     whitespace-aligned tables pdfplumber's default detector misses
     (e.g. the UBS "Our key figures" summary table on page 11).
"""

import json
import re
from pathlib import Path

import pdfplumber
import tqdm

from ingest.models import CompanyMetadata, PageData, PageSegment

# A numeric token: optional negative, optional parens (for negatives), digits with
# optional thousands separators, optional decimal, optional percent sign.
# Also accepts a bare en/em-dash (–, —) — Roche etc. use these as "not applicable"
# placeholders inside otherwise-numeric rows (e.g. Sales ... 14,104 – 58,716).
# Matches: 40,834 / 27,748 / 8.45 / 37.4 / (148) / -1.2 / 95.0% / – / —
_NUM = r"(?:-?\(?[\d][\d,]*(?:\.\d+)?\)?%?|[–—])"

# A "tabular" line: some leading non-numeric prefix (the row label) followed by
# 2+ trailing numeric tokens separated by whitespace.
_TABULAR = re.compile(rf"^\s*\S.*?\s+(?:{_NUM}\s+){{1,}}{_NUM}\s*$")


def parse_metadata(file_path: str) -> dict[str, CompanyMetadata]:
    """Parse metadata from a file path."""
    with open(file_path, "r", encoding="utf-8") as f:
        company_metadata = json.load(f)

    return {
        folder: CompanyMetadata(**data) for folder, data in company_metadata.items()
    }


def _table_to_markdown(table: list[list[str | None]]) -> str:
    """Serialize a pdfplumber-extracted table to a GFM markdown table.

    Returns '' for tables that aren't worth serializing:
      - empty / all-blank
      - single-column (almost always callouts or per-column detection artifacts
        where the row labels were missed — the linear text captures them better)
      - mostly empty cells (>60% blank — noisy grid detections)
    """
    if not table or not table[0]:
        return ""

    def clean(cell: str | None) -> str:
        if cell is None:
            return ""
        return " ".join(cell.split()).replace("|", "\\|")

    rows = [[clean(c) for c in row] for row in table]
    width = max(len(r) for r in rows)
    if width < 2:
        return ""

    rows = [r + [""] * (width - len(r)) for r in rows]
    total = sum(len(r) for r in rows)
    filled = sum(1 for r in rows for c in r if c)
    if filled == 0 or filled / total < 0.4:
        return ""

    header, body = rows[0], rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * width) + "|",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


_GAP_TOLERANCE = 3  # cluster tabular lines within this many lines of each other
_HEADER_LOOKBACK = 3  # how many short header lines to absorb above a table run


def _is_short_header_line(line: str) -> bool:
    """A line is considered 'header-like' (safe to pull into a table region)
    when it's short and not a sentence.
    """
    s = line.strip()
    if not s:
        return True
    if len(s) > 100:
        return False
    if s.endswith(".") and len(s) > 40:
        return False
    return True


def _find_table_regions(lines: list[str]) -> list[tuple[int, int]]:
    """Cluster tabular lines into table regions.

    Two tabular lines within _GAP_TOLERANCE lines of each other join the same
    region — this handles intra-table noise like partial-data rows
    (e.g. UBS "Negative goodwill 27,748" with only one comparison year) or
    subsection headers ("Profitability and growth", "Resources"). The region
    start is then extended upward through up to _HEADER_LOOKBACK short
    non-sentence lines to capture column headers / sub-titles.

    Returns half-open (start, end) indices for groups with >=3 tabular lines.
    """
    tabular = [i for i, ln in enumerate(lines) if _TABULAR.match(ln)]
    if not tabular:
        return []

    groups: list[list[int]] = [[tabular[0]]]
    for idx in tabular[1:]:
        if idx - groups[-1][-1] <= _GAP_TOLERANCE:
            groups[-1].append(idx)
        else:
            groups.append([idx])

    regions: list[tuple[int, int]] = []
    for group in groups:
        if len(group) < 3:
            continue
        start = group[0]
        for k in range(group[0] - 1, max(-1, group[0] - 1 - _HEADER_LOOKBACK), -1):
            if _is_short_header_line(lines[k]):
                start = k
            else:
                break
        end = group[-1] + 1
        regions.append((start, end))
    return regions


def _section_title(lines: list[str], before_idx: int) -> str | None:
    """Find a short title-like line within ~5 lines above `before_idx`."""
    for k in range(before_idx - 1, max(-1, before_idx - 6), -1):
        s = lines[k].strip()
        if not s:
            continue
        if len(s) <= 80 and not s.endswith("."):
            return s
        return None
    return None


def _segments_from_text(linear_text: str) -> list[PageSegment]:
    """Split linear page text into ordered prose + table segments."""
    if not linear_text:
        return []
    lines = linear_text.split("\n")
    regions = _find_table_regions(lines)
    if not regions:
        return [PageSegment(kind="prose", text=linear_text)]

    segments: list[PageSegment] = []
    cursor = 0
    for start, end in regions:
        if start > cursor:
            prose = "\n".join(lines[cursor:start]).strip()
            if prose:
                segments.append(PageSegment(kind="prose", text=prose))
        table_text = "\n".join(lines[start:end]).strip()
        if table_text:
            segments.append(
                PageSegment(
                    kind="table",
                    text=table_text,
                    section_title=_section_title(lines, start),
                )
            )
        cursor = end
    tail = "\n".join(lines[cursor:]).strip()
    if tail:
        segments.append(PageSegment(kind="prose", text=tail))
    return segments


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

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in tqdm.tqdm(
                enumerate(pdf.pages),
                total=len(pdf.pages),
                desc=f"Processing {pdf_path.name}",
            ):
                try:
                    linear_text = (page.extract_text() or "").strip()
                    detected_tables = page.extract_tables() or []
                except Exception as e:
                    print(
                        f"  WARN: failed page {i + 1} of {pdf_path.name}: {e}"
                    )
                    continue

                if not linear_text and not detected_tables:
                    continue

                segments = _segments_from_text(linear_text)
                for table in detected_tables:
                    md = _table_to_markdown(table)
                    if md:
                        segments.append(PageSegment(kind="table", text=md))

                if not segments:
                    continue

                pages.append(
                    PageData(
                        ticker=company_metadata.ticker,
                        company_name=company_metadata.name,
                        sector=company_metadata.sector,
                        exchange=company_metadata.exchange,
                        year=year,
                        page_number=i + 1,
                        text=linear_text,
                        segments=segments,
                    )
                )

    return pages


if __name__ == "__main__":
    parse_pdf("data/company_metadata.json")
    breakpoint()
