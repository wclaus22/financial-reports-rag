"""tests for the Generator class."""

from unittest.mock import MagicMock

from app.generation import SYSTEM_PROMPT, Generator
from app.models import RetrievedHit
from app.safety import GenerationSafety, NoSafetyMechanism


def make_anthropic_response(text_blocks: list[str]) -> MagicMock:
    """Build a fake anthropic response whose content is a list of text blocks."""
    content = []
    for t in text_blocks:
        block = MagicMock()
        block.type = "text"
        block.text = t
        content.append(block)
    response = MagicMock()
    response.content = content
    return response


def make_generator(
    response_text: list[str] | None = None,
    safety: GenerationSafety | None = None,
) -> tuple[Generator, MagicMock]:
    """Build a Generator with a mocked anthropic client and real NoSafetyMechanism by default."""
    anthropic_client = MagicMock()
    anthropic_client.messages.create.return_value = make_anthropic_response(
        response_text or ["an answer"]
    )

    safety = safety or GenerationSafety(NoSafetyMechanism())
    return Generator(safety=safety, anthropic_client=anthropic_client), anthropic_client


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


def test_generate_answer_returns_concatenated_text_blocks():
    generator, _ = make_generator(response_text=["Hello ", "world."])

    answer, _ = generator.generate_answer("q", [make_hit()])

    assert answer == "Hello world."


def test_generate_answer_skips_non_text_blocks():
    anthropic_client = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "kept"
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.text = "dropped"
    response = MagicMock()
    response.content = [text_block, tool_block]
    anthropic_client.messages.create.return_value = response
    generator = Generator(
        safety=GenerationSafety(NoSafetyMechanism()),
        anthropic_client=anthropic_client,
    )

    answer, _ = generator.generate_answer("q", [make_hit()])

    assert answer == "kept"


def test_generate_answer_uses_configured_model(monkeypatch):
    from app import generation

    monkeypatch.setattr(generation.settings, "llm_model", "claude-test-model")
    generator, anthropic_client = make_generator()

    generator.generate_answer("q", [make_hit()])

    assert anthropic_client.messages.create.call_args.kwargs["model"] == (
        "claude-test-model"
    )


def test_generate_answer_passes_system_prompt():
    generator, anthropic_client = make_generator()

    generator.generate_answer("q", [make_hit()])

    assert anthropic_client.messages.create.call_args.kwargs["system"] == SYSTEM_PROMPT


def test_generate_answer_includes_query_and_context_in_user_message():
    generator, anthropic_client = make_generator()
    hit = make_hit(ticker="ROG", year=2024, page_number=5, text="R&D up 7%.")

    generator.generate_answer("how did R&D evolve?", [hit])

    messages = anthropic_client.messages.create.call_args.kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    user_content = messages[0]["content"]
    assert "how did R&D evolve?" in user_content
    assert "[ROG 2024, p. 5]" in user_content
    assert "R&D up 7%." in user_content


def test_generate_answer_joins_multiple_blocks_with_separator():
    generator, anthropic_client = make_generator()
    hits = [
        make_hit(ticker="NESN", year=2023, page_number=12, text="first"),
        make_hit(ticker="ROG", year=2024, page_number=5, text="second"),
    ]

    generator.generate_answer("q", hits)

    user_content = anthropic_client.messages.create.call_args.kwargs["messages"][0][
        "content"
    ]
    assert "\n\n---\n\n" in user_content
    assert user_content.index("first") < user_content.index("second")


def test_generate_answer_returns_filtered_hits_from_safety():
    kept_hit = make_hit(chunk_id="kept")
    safety = MagicMock(spec=GenerationSafety)
    safety.add_hit.side_effect = lambda hit, filtered: filtered + [hit]
    safety.check_hits.return_value = [kept_hit]
    generator, _ = make_generator(safety=safety)

    _, filtered_hits = generator.generate_answer("q", [make_hit(), make_hit()])

    assert filtered_hits == [kept_hit]


def test_generate_answer_invokes_safety_for_each_hit():
    safety = MagicMock(spec=GenerationSafety)
    safety.add_hit.side_effect = lambda hit, filtered: filtered + [hit]
    safety.check_hits.side_effect = lambda filtered: filtered
    generator, _ = make_generator(safety=safety)
    hits = [make_hit(chunk_id="a"), make_hit(chunk_id="b"), make_hit(chunk_id="c")]

    generator.generate_answer("q", hits)

    assert safety.add_hit.call_count == 3
    safety.check_hits.assert_called_once()


def test_system_prompt_contains_grounding_guardrail():
    """The simple path must carry the same per-source discipline as the agent
    so the A/B fabrication comparison isn't confounded by prompt differences."""
    lowered = SYSTEM_PROMPT.lower()
    assert "exact company and year" in lowered
    assert "never infer, interpolate" in lowered
    assert "say so explicitly" in lowered
    assert "[ticker year, p. page]" in lowered


def test_generate_answer_with_no_hits_still_calls_llm():
    generator, anthropic_client = make_generator()

    answer, filtered_hits = generator.generate_answer("q", [])

    assert filtered_hits == []
    anthropic_client.messages.create.assert_called_once()
    assert answer == "an answer"
