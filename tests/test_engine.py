"""tests for the QueryEngine seam + build_engine factory."""

from unittest.mock import MagicMock

import pytest

from app.engine import AgentEngine, SimpleEngine, build_engine
from app.pipeline import QueryResult


def _result() -> QueryResult:
    return QueryResult(answer="a", retrieved=[], grounding=[])


# ---------------------------------------------------------------------------
# factory selection
# ---------------------------------------------------------------------------


def test_build_engine_agentic_returns_agent_engine():
    engine = build_engine(agentic=True, retriever=MagicMock())

    assert isinstance(engine, AgentEngine)


def test_build_engine_non_agentic_returns_simple_engine():
    engine = build_engine(
        agentic=False, retriever=MagicMock(), generator=MagicMock()
    )

    assert isinstance(engine, SimpleEngine)


def test_build_engine_simple_without_generator_raises():
    with pytest.raises(ValueError):
        build_engine(agentic=False, retriever=MagicMock())


def test_build_engine_agentic_passes_client_to_agent():
    """The factory wires the retriever + client into the Agent it builds."""
    retriever, client = MagicMock(), MagicMock()

    engine = build_engine(agentic=True, retriever=retriever, anthropic_client=client)

    assert engine.agent.retriever is retriever
    assert engine.agent.client is client


# ---------------------------------------------------------------------------
# delegation
# ---------------------------------------------------------------------------


def test_simple_engine_delegates_to_run_query(monkeypatch):
    from app import engine as engine_module

    result = _result()
    fake_run_query = MagicMock(return_value=result)
    monkeypatch.setattr(engine_module, "run_query", fake_run_query)

    retriever, generator = MagicMock(), MagicMock()
    engine = SimpleEngine(retriever, generator)

    out = engine.run("question", top_k=3)

    assert out is result
    fake_run_query.assert_called_once_with("question", 3, retriever, generator)


def test_agent_engine_delegates_to_agent_run():
    result = _result()
    agent = MagicMock()
    agent.run.return_value = result
    engine = AgentEngine(agent)

    out = engine.run("question", top_k=4)

    assert out is result
    agent.run.assert_called_once_with("question", 4)
