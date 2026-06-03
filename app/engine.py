"""
run a retrieve->generate query pipeline using different retrieval schemes implemented through various engines
"""

from typing import Protocol

from app.agent import Agent
from app.generation import Generator
from app.pipeline import QueryResult, run_query
from app.retrieval import Retriever


class QueryEngine(Protocol):
    """Engine protocol for running a question -> query result pipeline"""

    def run(self, question: str, top_k: int) -> QueryResult: ...


class SimpleEngine:
    """baseline engine built by a single retriever and a generator"""

    def __init__(self, retriever: Retriever, generator: Generator) -> None:
        self.retriever = retriever
        self.generator = generator

    def run(self, question: str, top_k: int) -> QueryResult:
        return run_query(question, top_k, self.retriever, self.generator)


class AgentEngine:
    """agentic path: the hand-rolled tool-use loop decomposes the query."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent

    def run(self, question: str, top_k: int) -> QueryResult:
        return self.agent.run(question, top_k)


def build_engine(
    agentic: bool,
    retriever: Retriever,
    generator: Generator | None = None,
    anthropic_client=None,
) -> QueryEngine:
    """
    Construct the engine selected by `agentic`.
    """
    if agentic:
        return AgentEngine(Agent(retriever, anthropic_client))

    if generator is None:
        raise ValueError("SimpleEngine requires a generator")
    return SimpleEngine(retriever, generator)
