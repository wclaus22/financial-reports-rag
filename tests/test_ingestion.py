"""tests for the ingestion pipeline: chunking, embedding, indexing, metadata parsing."""

import json
from unittest.mock import MagicMock

from ingest.chunk import chunk_pages
from ingest.embed import embed_batch
from ingest.embed_and_index import index_chunks
from ingest.models import Chunk, CompanyMetadata, PageData
from ingest.parse import parse_metadata


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
) -> PageData:
    return PageData(
        ticker=ticker,
        company_name="Nestle",
        sector="Food",
        exchange="SIX",
        year=year,
        page_number=page_number,
        text=text,
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
