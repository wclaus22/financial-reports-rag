"""
Shared retrieve->generate orchestration.
"""

from dataclasses import dataclass

from app.generation import Generator
from app.models import RetrievedHit
from app.retrieval import Retriever

NO_HITS_ANSWER = "No relevant excerpts found in the corpus."


@dataclass
class QueryResult:
    """Outcome of one query.

    retrieved: the raw retriever slate (pre-safety) -- what was fetched. This is
        what retrieval metrics should be scored against.
    grounding: the filtered hits the generator actually used to answer. This is
        what the API exposes as `sources` and what the LLM judge should check.

    On the no-hits short-circuit, both lists are empty and `answer` is
    NO_HITS_ANSWER.
    """

    answer: str
    retrieved: list[RetrievedHit]
    grounding: list[RetrievedHit]


def run_query(
    question: str,
    top_k: int,
    retriever: Retriever,
    generator: Generator,
) -> QueryResult:
    """Retrieve, then (unless there are no hits) generate.

    The no-hits short-circuit returns NO_HITS_ANSWER WITHOUT calling the
    generator -- mirroring the /query behaviour exactly and avoiding a wasted
    LLM call. Note this is distinct from "hits found but the safety layer
    filtered them all": in that case the generator IS called and `grounding`
    may still be empty while `answer` is a real model response.
    """
    hits = retriever.retrieve(question, top_k)
    if not hits:
        return QueryResult(answer=NO_HITS_ANSWER, retrieved=[], grounding=[])

    answer, filtered_hits = generator.generate_answer(question, hits)
    return QueryResult(answer=answer, retrieved=hits, grounding=filtered_hits)
