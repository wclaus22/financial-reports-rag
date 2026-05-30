"""tests for the ingestion pipeline: chunking, embedding, indexing, metadata parsing."""

import json
from unittest.mock import MagicMock

from ingest.chunk import chunk_pages
from ingest.embed import embed_batch
from ingest.embed_and_index import index_chunks
from ingest.models import Chunk, CompanyMetadata, PageData, PageSegment
from ingest.parse import (
    _find_table_regions,
    _segments_from_text,
    _table_to_markdown,
    parse_metadata,
)


def make_chunk(
    chunk_id: str = "NESN_2023_page1_chunk1",
    text: str = "Net sales grew 7%.",
    ticker: str = "NESN",
    year: int = 2023,
    page_number: int = 1,
) -> Chunk:
    return Chunk(
        ticker=ticker,
        company_name="Nestle",
        sector="Food",
        exchange="SIX",
        year=year,
        page_number=page_number,
        chunk_id=chunk_id,
        text=text,
    )


def make_page(
    text: str = "Net sales grew 7%.",
    ticker: str = "NESN",
    year: int = 2023,
    page_number: int = 1,
    segments: list[PageSegment] | None = None,
) -> PageData:
    return PageData(
        ticker=ticker,
        company_name="Nestle",
        sector="Food",
        exchange="SIX",
        year=year,
        page_number=page_number,
        text=text,
        segments=segments if segments is not None else [PageSegment(kind="prose", text=text)],
    )


def make_voyage_and_collection(
    embeddings: list[list[float]] | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Build mocked voyage client + chroma collection for index_chunks tests."""
    voyage = MagicMock()
    voyage.embed.return_value.embeddings = embeddings or [[0.1, 0.2, 0.3]]
    collection = MagicMock()
    return voyage, collection


# ---------------------------------------------------------------------------
# index_chunks
# ---------------------------------------------------------------------------


def test_index_chunks_passes_documents_to_collection():
    """Regression test for the bug where documents= was omitted from collection.add."""
    chunks = [make_chunk(chunk_id="a", text="alpha"), make_chunk(chunk_id="b", text="beta")]
    voyage, collection = make_voyage_and_collection(embeddings=[[0.1], [0.2]])

    index_chunks(chunks, voyage, collection, batch_size=10)

    assert collection.add.call_args.kwargs["documents"] == ["alpha", "beta"]


def test_index_chunks_passes_ids_and_metadatas():
    chunks = [
        make_chunk(chunk_id="a", ticker="NESN", year=2023, page_number=1),
        make_chunk(chunk_id="b", ticker="ROG", year=2024, page_number=5),
    ]
    voyage, collection = make_voyage_and_collection(embeddings=[[0.1], [0.2]])

    index_chunks(chunks, voyage, collection, batch_size=10)

    kwargs = collection.add.call_args.kwargs
    assert kwargs["ids"] == ["a", "b"]
    assert kwargs["metadatas"][0]["ticker"] == "NESN"
    assert kwargs["metadatas"][0]["year"] == 2023
    assert kwargs["metadatas"][0]["page_number"] == 1
    assert kwargs["metadatas"][1]["ticker"] == "ROG"
    assert kwargs["metadatas"][1]["year"] == 2024
    assert set(kwargs["metadatas"][0].keys()) == {
        "ticker",
        "company_name",
        "sector",
        "exchange",
        "year",
        "page_number",
    }


def test_index_chunks_passes_embeddings_from_voyage():
    chunks = [make_chunk(chunk_id="a"), make_chunk(chunk_id="b")]
    embeddings = [[0.11, 0.22], [0.33, 0.44]]
    voyage, collection = make_voyage_and_collection(embeddings=embeddings)

    index_chunks(chunks, voyage, collection, batch_size=10)

    assert collection.add.call_args.kwargs["embeddings"] == embeddings


def test_index_chunks_batches_by_batch_size():
    chunks = [make_chunk(chunk_id=f"c{i}", text=f"t{i}") for i in range(5)]
    voyage = MagicMock()
    voyage.embed.return_value.embeddings = [[0.0]] * 2  # default, overridden per call
    voyage.embed.side_effect = [
        MagicMock(embeddings=[[0.0], [0.1]]),
        MagicMock(embeddings=[[0.2], [0.3]]),
        MagicMock(embeddings=[[0.4]]),
    ]
    collection = MagicMock()

    index_chunks(chunks, voyage, collection, batch_size=2)

    assert collection.add.call_count == 3
    first_call_docs = collection.add.call_args_list[0].kwargs["documents"]
    last_call_docs = collection.add.call_args_list[2].kwargs["documents"]
    assert first_call_docs == ["t0", "t1"]
    assert last_call_docs == ["t4"]


def test_index_chunks_calls_voyage_with_document_input_type(monkeypatch):
    """embed_batch must use input_type='document' — guard against drift to 'query'."""
    from ingest import embed_and_index

    captured = {}

    def fake_embed_batch(client, texts):
        captured["client"] = client
        captured["texts"] = texts
        return [[0.0] for _ in texts]

    monkeypatch.setattr(embed_and_index, "embed_batch", fake_embed_batch)

    chunks = [make_chunk(text="hello")]
    voyage, collection = make_voyage_and_collection()

    index_chunks(chunks, voyage, collection, batch_size=10)

    assert captured["texts"] == ["hello"]
    assert captured["client"] is voyage


def test_index_chunks_empty_chunks_is_noop():
    voyage, collection = make_voyage_and_collection()

    index_chunks([], voyage, collection, batch_size=10)

    voyage.embed.assert_not_called()
    collection.add.assert_not_called()


# ---------------------------------------------------------------------------
# chunk_pages
# ---------------------------------------------------------------------------


def test_chunk_pages_preserves_page_metadata():
    page = make_page(
        text="some short text",
        ticker="ROG",
        year=2024,
        page_number=7,
    )

    chunks = chunk_pages([page], chunk_size=1000, chunk_overlap=100)

    assert len(chunks) >= 1
    for c in chunks:
        assert c.ticker == "ROG"
        assert c.company_name == "Nestle"
        assert c.sector == "Food"
        assert c.exchange == "SIX"
        assert c.year == 2024
        assert c.page_number == 7


def test_chunk_pages_generates_unique_ids_with_expected_format():
    long_text = ("paragraph one. " * 50 + "\n\n" + "paragraph two. " * 50).strip()
    page = make_page(text=long_text, ticker="NESN", year=2023, page_number=3)

    chunks = chunk_pages([page], chunk_size=100, chunk_overlap=10)

    assert len(chunks) > 1
    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)
    for i, c in enumerate(chunks):
        assert c.chunk_id == f"NESN_2023_page3_chunk{i + 1}"


def test_chunk_pages_splits_long_text_into_multiple_chunks():
    long_text = "word " * 500
    page = make_page(text=long_text)

    chunks = chunk_pages([page], chunk_size=100, chunk_overlap=10)

    assert len(chunks) > 1


def test_chunk_pages_prepends_context_prefix():
    page = make_page(text="Net sales grew 7%.", ticker="NESN", year=2023, page_number=1)

    chunks = chunk_pages([page])

    assert len(chunks) >= 1
    assert chunks[0].text.startswith(
        "Nestle (NESN) — Annual Report 2023 — p.1"
    )
    # the body still has the original content after the prefix
    assert "Net sales grew 7%." in chunks[0].text


def test_chunk_pages_prefix_includes_page_heading_when_present():
    page = make_page(
        text="Our key figures\n\nSome body text describing the figures.",
        page_number=11,
    )

    chunks = chunk_pages([page])

    assert chunks[0].text.startswith(
        "Nestle (NESN) — Annual Report 2023 — p.11 — Our key figures"
    )


def test_chunk_pages_table_segment_emits_single_chunk_even_if_large():
    # 4000-char table body — well over chunk_size — should still produce ONE chunk.
    table_body = "Total revenues 40,834 34,563 35,393\n" * 100
    page = make_page(
        text="ignored",
        segments=[
            PageSegment(kind="table", text=table_body, section_title="Our key figures")
        ],
    )

    chunks = chunk_pages([page], chunk_size=1000, chunk_overlap=100)

    assert len(chunks) == 1
    assert chunks[0].text.startswith(
        "Nestle (NESN) — Annual Report 2023 — p.1"
    )
    assert "Table: Our key figures" in chunks[0].text
    # full table body preserved verbatim
    assert table_body.strip() in chunks[0].text


def test_chunk_pages_prose_and_table_segments_share_page_chunk_numbering():
    page = make_page(
        text="ignored",
        segments=[
            PageSegment(kind="prose", text="Lead-in paragraph for the table."),
            PageSegment(kind="table", text="Total revenues 1 2 3"),
        ],
    )

    chunks = chunk_pages([page])

    assert [c.chunk_id for c in chunks] == [
        "NESN_2023_page1_chunk1",
        "NESN_2023_page1_chunk2",
    ]


# ---------------------------------------------------------------------------
# parse: segment detection
# ---------------------------------------------------------------------------


def test_find_table_regions_detects_whitespace_aligned_table():
    text = (
        "Our key figures\n"
        "31.12.23 31.12.22 31.12.21\n"
        "Total revenues 40,834 34,563 35,393\n"
        "Operating profit 28,739 9,604 9,484\n"
        "Net profit 27,849 7,630 7,457\n"
        "Footnote text below the table.\n"
    )
    lines = text.split("\n")

    regions = _find_table_regions(lines)

    assert len(regions) == 1
    start, end = regions[0]
    region_lines = lines[start:end]
    # The region pulls in the column header and section heading above the rows.
    assert any("31.12.23 31.12.22 31.12.21" in ln for ln in region_lines)
    assert any("Our key figures" in ln for ln in region_lines)
    # And covers all 3 tabular rows.
    assert any("Total revenues 40,834" in ln for ln in region_lines)
    assert any("Net profit 27,849" in ln for ln in region_lines)
    # The footnote prose stays out.
    assert not any("Footnote text" in ln for ln in region_lines)


def test_find_table_regions_returns_nothing_for_pure_prose():
    text = (
        "This page is about strategy. We continued to invest in our long-term "
        "growth initiatives, focused on sustainability and innovation."
    )
    lines = text.split("\n")

    regions = _find_table_regions(lines)

    assert regions == []


def test_segments_from_text_alternates_prose_and_table():
    text = (
        "Introduction paragraph that frames the data below.\n"
        "Our key figures\n"
        "Total revenues 40,834 34,563 35,393\n"
        "Operating profit 28,739 9,604 9,484\n"
        "Net profit 27,849 7,630 7,457\n"
        "Closing commentary after the table.\n"
    )

    segments = _segments_from_text(text)

    kinds = [s.kind for s in segments]
    assert kinds == ["prose", "table", "prose"]
    assert "Introduction paragraph" in segments[0].text
    assert "Total revenues" in segments[1].text
    # the heading line directly above the tabular rows gets pulled into the
    # table region (it might be a column header or a section title — either way
    # it belongs with the table content).
    assert "Our key figures" in segments[1].text
    assert "Closing commentary" in segments[2].text


def test_segments_from_text_pulls_heading_into_table_body():
    """When a short heading sits right above the tabular rows, the clustering
    pulls it into the table body — keeping the heading semantically together
    with the rows it describes."""
    text = (
        "Some long paragraph of prose that sets up the financial review and "
        "talks about strategy in some detail across several clauses.\n"
        "Our key figures\n"
        "Total revenues 40,834 34,563 35,393\n"
        "Operating profit 28,739 9,604 9,484\n"
        "Net profit 27,849 7,630 7,457\n"
    )

    segments = _segments_from_text(text)

    table_seg = next(s for s in segments if s.kind == "table")
    assert "Our key figures" in table_seg.text
    assert "Total revenues 40,834" in table_seg.text


def test_segments_from_text_returns_single_prose_when_no_table():
    text = "Just some plain prose with no tabular content."

    segments = _segments_from_text(text)

    assert len(segments) == 1
    assert segments[0].kind == "prose"
    assert segments[0].text == text


# ---------------------------------------------------------------------------
# parse: markdown table serialization
# ---------------------------------------------------------------------------


def test_table_to_markdown_basic():
    table = [["Year", "Rev", "Profit"], ["2023", "100", "10"], ["2022", "90", "8"]]

    md = _table_to_markdown(table)

    assert "| Year | Rev | Profit |" in md
    assert "|---|---|---|" in md
    assert "| 2023 | 100 | 10 |" in md


def test_table_to_markdown_skips_single_column():
    assert _table_to_markdown([["Header"], ["Value 1"], ["Value 2"]]) == ""


def test_table_to_markdown_skips_mostly_empty():
    table = [["A", "B", "C"], ["", "", ""], ["", "", ""], ["", "x", ""]]
    assert _table_to_markdown(table) == ""


def test_table_to_markdown_handles_none_cells():
    table = [["A", "B"], [None, "x"], ["y", None]]

    md = _table_to_markdown(table)

    assert "|  | x |" in md
    assert "| y |  |" in md


# ---------------------------------------------------------------------------
# parse_metadata
# ---------------------------------------------------------------------------


def test_parse_metadata_returns_company_metadata_keyed_by_folder(tmp_path):
    metadata_file = tmp_path / "company_metadata.json"
    metadata_file.write_text(
        json.dumps(
            {
                "nestle": {
                    "name": "Nestle",
                    "ticker": "NESN",
                    "sector": "Food",
                    "exchange": "SIX",
                },
                "roche": {
                    "name": "Roche",
                    "ticker": "ROG",
                    "sector": "Health Care",
                    "exchange": "SIX",
                },
            }
        )
    )

    result = parse_metadata(str(metadata_file))

    assert set(result.keys()) == {"nestle", "roche"}
    assert isinstance(result["nestle"], CompanyMetadata)
    assert result["nestle"].ticker == "NESN"
    assert result["roche"].sector == "Health Care"


# ---------------------------------------------------------------------------
# embed_batch
# ---------------------------------------------------------------------------


def test_embed_batch_uses_document_input_type():
    voyage = MagicMock()
    voyage.embed.return_value.embeddings = [[0.1, 0.2]]

    embed_batch(voyage, ["hello"])

    kwargs = voyage.embed.call_args.kwargs
    assert kwargs["input_type"] == "document"


def test_embed_batch_returns_embeddings_from_voyage():
    voyage = MagicMock()
    voyage.embed.return_value.embeddings = [[0.1, 0.2], [0.3, 0.4]]

    result = embed_batch(voyage, ["a", "b"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    args = voyage.embed.call_args.args
    assert args[0] == ["a", "b"]
