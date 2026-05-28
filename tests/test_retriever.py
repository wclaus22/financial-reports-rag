"""tests for the Retriever class"""

from unittest.mock import MagicMock

from app.models import RetrievedHit
from app.retrieval import Retriever


def make_retriever(
    query_result: dict, embedding: list[float] | None = None
) -> tuple[Retriever, MagicMock, MagicMock]:
    """Build a Retriever with mocked voyage client + chroma collection."""
    voyage = MagicMock()
    voyage.embed.return_value.embeddings = [embedding or [0.1, 0.2, 0.3]]

    collection = MagicMock()
    collection.query.return_value = query_result

    return Retriever(voyage_client=voyage, collection=collection), voyage, collection


EMPTY_RESULT = {
    "ids": [[]],
    "documents": [[]],
    "metadatas": [[]],
    "distances": [[]],
}


def test_retrieve_returns_typed_hits():
    result = {
        "ids": [["chunk-1", "chunk-2"]],
        "documents": [["text one", "text two"]],
        "metadatas": [
            [
                {
                    "ticker": "NESN",
                    "year": 2023,
                    "page_number": 12,
                    "sector": "Food",
                    "exchange": "SIX",
                    "company_name": "Nestle",
                },
                {
                    "ticker": "ROG",
                    "year": 2024,
                    "page_number": 5,
                    "sector": "Health Care",
                    "exchange": "SIX",
                    "company_name": "Roche",
                },
            ]
        ],
        "distances": [[0.12, 0.34]],
    }
    retriever, _, _ = make_retriever(result)

    hits = retriever.retrieve("revenue growth?")

    assert len(hits) == 2
    assert hits[0] == RetrievedHit(
        ticker="NESN",
        year=2023,
        page_number=12,
        sector="Food",
        exchange="SIX",
        company_name="Nestle",
        text="text one",
        distance=0.12,
        chunk_id="chunk-1",
    )
    assert hits[1].chunk_id == "chunk-2"
    assert hits[1].distance == 0.34


def test_retrieve_empty_results_returns_empty_list():
    retriever, _, _ = make_retriever(EMPTY_RESULT)

    assert retriever.retrieve("anything") == []


def test_retrieve_passes_explicit_top_k_to_chroma():
    retriever, _, collection = make_retriever(EMPTY_RESULT)

    retriever.retrieve("q", top_k=3)

    assert collection.query.call_args.kwargs["n_results"] == 3


def test_retrieve_falls_back_to_settings_top_k(monkeypatch):
    from app import retrieval

    monkeypatch.setattr(retrieval.settings, "top_k", 7)
    retriever, _, collection = make_retriever(EMPTY_RESULT)

    retriever.retrieve("q")

    assert collection.query.call_args.kwargs["n_results"] == 7


def test_retrieve_embeds_query_with_voyage():
    retriever, voyage, _ = make_retriever(EMPTY_RESULT)

    retriever.retrieve("net sales of nestle in 2023?")

    voyage.embed.assert_called_once()
    args, kwargs = voyage.embed.call_args
    assert args[0] == ["net sales of nestle in 2023?"]
    assert kwargs["input_type"] == "query"


def test_retrieve_forwards_embedding_to_chroma():
    embedding = [0.5, 0.6, 0.7]
    retriever, _, collection = make_retriever(EMPTY_RESULT, embedding=embedding)

    retriever.retrieve("q")

    assert collection.query.call_args.kwargs["query_embeddings"] == [embedding]
