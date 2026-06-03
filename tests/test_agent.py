"""tests for the hand-rolled Agent tool-use loop."""

from unittest.mock import MagicMock

from app.agent import Agent
from app.models import RetrievedHit


# ---------------------------------------------------------------------------
# fake anthropic content blocks / responses
# ---------------------------------------------------------------------------


def _text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _tool_use_block(block_id: str, tool_input: dict, name: str = "search") -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = block_id
    block.name = name
    block.input = tool_input
    return block


def _response(content: list, stop_reason: str) -> MagicMock:
    response = MagicMock()
    response.content = content
    response.stop_reason = stop_reason
    return response


def _tool_turn(*blocks) -> MagicMock:
    return _response(list(blocks), stop_reason="tool_use")


def _text_turn(text: str) -> MagicMock:
    return _response([_text_block(text)], stop_reason="end_turn")


def make_agent(
    responses: list, search_hits=None
) -> tuple[Agent, MagicMock, MagicMock]:
    """Build an Agent with a scripted anthropic client + mocked retriever.

    `responses` is consumed one-per-`messages.create` call. `search_hits` is
    either a fixed list returned by every `retriever.search`, or a list of
    lists used as `side_effect` (one per search call).
    """
    client = MagicMock()
    client.messages.create.side_effect = responses

    retriever = MagicMock()
    if isinstance(search_hits, list) and search_hits and isinstance(search_hits[0], list):
        retriever.search.side_effect = search_hits
    else:
        retriever.search.return_value = search_hits or []

    return Agent(retriever=retriever, anthropic_client=client), client, retriever


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


# ---------------------------------------------------------------------------
# happy path: search then answer
# ---------------------------------------------------------------------------


def test_run_executes_searches_then_returns_answer():
    """Parallel two-search turn, then a text answer on the next turn."""
    agent, client, retriever = make_agent(
        responses=[
            _tool_turn(
                _tool_use_block("t1", {"query": "roche revenue", "ticker": "ROG"}),
                _tool_use_block("t2", {"query": "nestle revenue", "ticker": "NESN"}),
            ),
            _text_turn("Roche and Nestle both grew."),
        ],
        search_hits=[
            [make_hit(ticker="ROG", chunk_id="rog-1")],
            [make_hit(ticker="NESN", chunk_id="nesn-1")],
        ],
    )

    result = agent.run("compare roche and nestle revenue")

    assert result.answer == "Roche and Nestle both grew."
    assert client.messages.create.call_count == 2
    assert retriever.search.call_count == 2


def test_run_answers_immediately_without_searching():
    """If the first turn is already text, no search happens and the union is empty."""
    agent, client, retriever = make_agent(responses=[_text_turn("I cannot answer.")])

    result = agent.run("out of corpus question")

    assert result.answer == "I cannot answer."
    assert client.messages.create.call_count == 1
    retriever.search.assert_not_called()
    assert result.retrieved == []
    assert result.grounding == []


# ---------------------------------------------------------------------------
# union / grounding semantics
# ---------------------------------------------------------------------------


def test_retrieved_is_deduped_union_in_fetch_order():
    """Overlapping chunk_ids across searches appear once, in first-seen order."""
    agent, _, _ = make_agent(
        responses=[
            _tool_turn(
                _tool_use_block("t1", {"query": "q", "year": 2023}),
                _tool_use_block("t2", {"query": "q", "year": 2024}),
            ),
            _text_turn("done"),
        ],
        search_hits=[
            [make_hit(chunk_id="a"), make_hit(chunk_id="b")],
            [make_hit(chunk_id="b"), make_hit(chunk_id="c")],  # b overlaps
        ],
    )

    result = agent.run("q")

    assert [h.chunk_id for h in result.retrieved] == ["a", "b", "c"]


def test_grounding_equals_retrieved():
    """The fabrication judge must see exactly what the model saw."""
    agent, _, _ = make_agent(
        responses=[
            _tool_turn(_tool_use_block("t1", {"query": "q"})),
            _text_turn("done"),
        ],
        search_hits=[make_hit(chunk_id="a")],
    )

    result = agent.run("q")

    assert result.grounding == result.retrieved


# ---------------------------------------------------------------------------
# message-shape contract with the API
# ---------------------------------------------------------------------------


def test_every_tool_use_gets_matching_tool_result():
    """The user turn after a tool-use turn must carry one tool_result per
    tool_use block, with matching tool_use_ids."""
    agent, client, _ = make_agent(
        responses=[
            _tool_turn(
                _tool_use_block("t1", {"query": "q1"}),
                _tool_use_block("t2", {"query": "q2"}),
            ),
            _text_turn("done"),
        ],
        search_hits=[make_hit()],
    )

    agent.run("q")

    # messages is mutated in place, so locate the tool_result turn (the only
    # user turn whose content is a list of blocks) rather than indexing [-1].
    all_messages = client.messages.create.call_args_list[-1].kwargs["messages"]
    result_turns = [
        m for m in all_messages if m["role"] == "user" and isinstance(m["content"], list)
    ]
    assert len(result_turns) == 1
    blocks = result_turns[0]["content"]
    assert [b["type"] for b in blocks] == ["tool_result", "tool_result"]
    assert {b["tool_use_id"] for b in blocks} == {"t1", "t2"}


def test_assistant_turn_appended_verbatim():
    """The raw assistant content blocks are echoed back into the history."""
    tool_turn = _tool_turn(_tool_use_block("t1", {"query": "q"}))
    agent, client, _ = make_agent(
        responses=[tool_turn, _text_turn("done")],
        search_hits=[make_hit()],
    )

    agent.run("q")

    second_call_messages = client.messages.create.call_args_list[1].kwargs["messages"]
    assert second_call_messages[1] == {
        "role": "assistant",
        "content": tool_turn.content,
    }


def test_zero_hit_search_renders_placeholder():
    agent, client, _ = make_agent(
        responses=[
            _tool_turn(_tool_use_block("t1", {"query": "q", "ticker": "ROG"})),
            _text_turn("done"),
        ],
        search_hits=[],
    )

    agent.run("q")

    all_messages = client.messages.create.call_args_list[-1].kwargs["messages"]
    result_turn = next(
        m for m in all_messages if m["role"] == "user" and isinstance(m["content"], list)
    )
    assert result_turn["content"][0]["content"] == (
        "No relevant excerpts found for this search."
    )


# ---------------------------------------------------------------------------
# filters / config wiring
# ---------------------------------------------------------------------------


def test_filters_forwarded_to_search():
    agent, _, retriever = make_agent(
        responses=[
            _tool_turn(
                _tool_use_block("t1", {"query": "revenue", "ticker": "ROG", "year": 2024})
            ),
            _text_turn("done"),
        ],
        search_hits=[make_hit()],
    )

    agent.run("q", top_k=3)

    kwargs = retriever.search.call_args.kwargs
    assert kwargs["query"] == "revenue"
    assert kwargs["ticker"] == "ROG"
    assert kwargs["year"] == 2024
    assert kwargs["top_k"] == 3


def test_uses_configured_agent_model(monkeypatch):
    from app import agent as agent_module

    monkeypatch.setattr(agent_module.settings, "agent_model", "claude-test-model")
    agent, client, _ = make_agent(responses=[_text_turn("done")])

    agent.run("q")

    assert client.messages.create.call_args.kwargs["model"] == "claude-test-model"


# ---------------------------------------------------------------------------
# termination guarantee
# ---------------------------------------------------------------------------


def test_iteration_cap_terminates_and_forces_final_answer(monkeypatch):
    """A model that always wants to search must still stop at the cap, and the
    final turn must disable tools to force a text answer."""
    from app import agent as agent_module

    monkeypatch.setattr(agent_module.settings, "agent_max_iterations", 3)
    always_search = _tool_turn(_tool_use_block("t1", {"query": "q"}))
    agent, client, _ = make_agent(
        responses=[always_search, always_search, always_search],
        search_hits=[make_hit()],
    )

    agent.run("q")

    # exactly the cap — no infinite loop
    assert client.messages.create.call_count == 3
    calls = client.messages.create.call_args_list
    # non-final turns allow tools; the final turn disables them
    assert calls[0].kwargs["tool_choice"] == {"type": "auto"}
    assert calls[1].kwargs["tool_choice"] == {"type": "auto"}
    assert calls[2].kwargs["tool_choice"] == {"type": "none"}
