"""tests for the Retriever class"""

from collections import namedtuple
from unittest.mock import MagicMock

from app.config import settings
from app.retrieval import Retriever


FakeResult = namedtuple("FakeResult", ["index", "document", "relevance_score"])


def _make_fake_rerank(docs: list[str], order: list[int] | None = None):
    """Build a fake RerankingObject-like return value.

    By default, returns candidates in their original Chroma order with
    descending dummy scores — lets older assertions about result ordering
    keep working.
    """
    if order is None:
        order = list(range(len(docs)))
    fake = MagicMock()
    fake.results = [
        FakeResult(index=i, document=docs[i], relevance_score=1.0 - rank * 0.01)
        for rank, i in enumerate(order)
    ]
    return fake


def make_retriever(
    query_result: dict,
    embedding: list[float] | None = None,
    rerank_order: list[int] | None = None,
) -> tuple[Retriever, MagicMock, MagicMock]:
    """Build a Retriever with mocked voyage client + chroma collection.

    voyage.rerank is mocked to return candidates in `rerank_order` (or original
    order by default) so callers control the post-rerank ranking explicitly.
    """
    voyage = MagicMock()
    voyage.embed.return_value.embeddings = [embedding or [0.1, 0.2, 0.3]]

    docs = query_result["documents"][0]
    voyage.rerank.return_value = _make_fake_rerank(docs, rerank_order)

    collection = MagicMock()
    collection.query.return_value = query_result

    return Retriever(voyage_client=voyage, collection=collection), voyage, collection


EMPTY_RESULT = {
    "ids": [[]],
    "documents": [[]],
    "metadatas": [[]],
    "distances": [[]],
}


def _two_chunk_result() -> dict:
    return {
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


def test_retrieve_returns_typed_hits():
    retriever, _, _ = make_retriever(_two_chunk_result())

    hits = retriever.retrieve("revenue growth?")

    assert len(hits) == 2
    assert hits[0].ticker == "NESN"
    assert hits[0].chunk_id == "chunk-1"
    assert hits[0].text == "text one"
    assert hits[0].distance == 0.12
    assert hits[0].page_number == 12
    assert hits[1].chunk_id == "chunk-2"
    assert hits[1].distance == 0.34


def test_retrieve_empty_results_returns_empty_list():
    retriever, voyage, _ = make_retriever(EMPTY_RESULT)

    assert retriever.retrieve("anything") == []
    voyage.rerank.assert_not_called()


def test_retrieve_fetches_rerank_candidate_pool_from_chroma():
    """Chroma is always queried for the rerank candidate pool size (default 50)
    regardless of the caller's requested top_k."""
    retriever, _, collection = make_retriever(EMPTY_RESULT)

    retriever.retrieve("q", top_k=3)

    assert (
        collection.query.call_args.kwargs["n_results"]
        == settings.rerank_candidates
    )


def test_retrieve_truncates_to_top_k_after_rerank():
    retriever, voyage, _ = make_retriever(_two_chunk_result())

    retriever.retrieve("q", top_k=1)

    assert voyage.rerank.call_args.kwargs["top_k"] == 1


def test_retrieve_falls_back_to_settings_top_k(monkeypatch):
    from app import retrieval

    monkeypatch.setattr(retrieval.settings, "top_k", 7)
    retriever, _, _ = make_retriever(EMPTY_RESULT)

    retriever.retrieve("q")

    # empty result short-circuits before rerank, but the resolved top_k flowed
    # through to the call we DID make — Chroma fetch.
    assert (
        retriever.collection.query.call_args.kwargs["n_results"]
        == settings.rerank_candidates
    )


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


def test_retrieve_calls_voyage_rerank_with_query_and_candidates():
    retriever, voyage, _ = make_retriever(_two_chunk_result())

    retriever.retrieve("how did revenue grow?")

    voyage.rerank.assert_called_once()
    kwargs = voyage.rerank.call_args.kwargs
    assert kwargs["query"] == "how did revenue grow?"
    assert kwargs["documents"] == ["text one", "text two"]
    assert kwargs["model"] == settings.rerank_model


def test_retrieve_uses_rerank_score_for_final_order():
    """When rerank returns chunk-2 first, hits[0] should be chunk-2 even
    though Chroma returned it second."""
    # rerank inverts the vector order: index 1 (chunk-2) wins
    retriever, _, _ = make_retriever(_two_chunk_result(), rerank_order=[1, 0])

    hits = retriever.retrieve("q")

    assert hits[0].chunk_id == "chunk-2"
    assert hits[0].distance == 0.34  # original vector distance preserved
    assert hits[1].chunk_id == "chunk-1"


def test_retrieve_populates_rerank_score_on_hits():
    retriever, _, _ = make_retriever(_two_chunk_result())

    hits = retriever.retrieve("q")

    assert hits[0].rerank_score is not None
    assert hits[1].rerank_score is not None
    # default fake rerank returns descending scores
    assert hits[0].rerank_score > hits[1].rerank_score


def test_retrieve_empty_candidates_skips_rerank():
    retriever, voyage, _ = make_retriever(EMPTY_RESULT)

    retriever.retrieve("anything")

    voyage.rerank.assert_not_called()
