"""tests for the shared retrieve->generate pipeline (app.pipeline.run_query)."""

from unittest.mock import MagicMock

from app.generation import Generator
from app.models import RetrievedHit
from app.pipeline import NO_HITS_ANSWER, run_query
from app.retrieval import Retriever


def make_hit(
    ticker: str = "NESN",
    year: int = 2023,
    page_number: int = 12,
    text: str = "Net sales grew 7%.",
    chunk_id: str = "chunk-1",
) -> RetrievedHit:
    return RetrievedHit(
        ticker=ticker,
        company_name="Nestle",
        sector="Food",
        exchange="SIX",
        year=year,
        page_number=page_number,
        text=text,
        distance=0.1,
        chunk_id=chunk_id,
    )


def make_collaborators(
    hits: list[RetrievedHit],
    answer: str = "an answer",
    grounding: list[RetrievedHit] | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Build mocked Retriever + Generator returning the given hits/answer."""
    retriever = MagicMock(spec=Retriever)
    retriever.retrieve.return_value = hits
    generator = MagicMock(spec=Generator)
    generator.generate_answer.return_value = (
        answer,
        hits if grounding is None else grounding,
    )
    return retriever, generator


def test_run_query_no_hits_short_circuits_without_generating():
    retriever, generator = make_collaborators(hits=[])

    result = run_query("q", 5, retriever, generator)

    assert result.answer == NO_HITS_ANSWER
    assert result.retrieved == []
    assert result.grounding == []
    generator.generate_answer.assert_not_called()


def test_run_query_hits_path_returns_answer_and_grounding():
    slate = [make_hit(chunk_id="a"), make_hit(chunk_id="b"), make_hit(chunk_id="c")]
    grounding = [slate[0], slate[1]]  # generator/safety kept a subset
    retriever, generator = make_collaborators(
        hits=slate, answer="the answer", grounding=grounding
    )

    result = run_query("q", 5, retriever, generator)

    assert result.answer == "the answer"
    assert result.retrieved == slate  # raw slate preserved
    assert result.grounding == grounding  # filtered subset surfaced independently
    assert result.retrieved is not result.grounding


def test_run_query_passes_question_and_top_k_to_retriever():
    retriever, generator = make_collaborators(hits=[make_hit()])

    run_query("q", 5, retriever, generator)

    retriever.retrieve.assert_called_once_with("q", 5)


def test_run_query_passes_question_and_hits_to_generator():
    slate = [make_hit(chunk_id="a"), make_hit(chunk_id="b")]
    retriever, generator = make_collaborators(hits=slate)

    run_query("q", 5, retriever, generator)

    generator.generate_answer.assert_called_once_with("q", slate)


def test_run_query_grounding_can_be_empty_with_nonempty_retrieved():
    """Hits found but safety filtered them all -> NOT the no-hits short-circuit."""
    slate = [make_hit()]
    retriever, generator = make_collaborators(
        hits=slate, answer="answer", grounding=[]
    )

    result = run_query("q", 5, retriever, generator)

    assert result.retrieved == slate
    assert result.grounding == []
    assert result.answer == "answer"
    generator.generate_answer.assert_called_once()


def test_no_hits_answer_constant_matches_served_string():
    assert NO_HITS_ANSWER == "No relevant excerpts found in the corpus."
