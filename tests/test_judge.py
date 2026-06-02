"""tests for the LLM judge (evals.judge)."""

from unittest.mock import MagicMock

import pytest

from app.config import settings
from app.models import RetrievedHit
from evals import judge as judge_module
from evals.dataset import EvalQuestion, GoldLocation
from evals.judge import Judge, judge_answer


def make_question(
    qid: str = "A001",
    question: str = "What was Roche's R&D in 2024?",
    expected_answer: str = "CHF 13.0 billion",
    expected_behavior: str = "answer",
) -> EvalQuestion:
    return EvalQuestion(
        id=qid,
        category="single_doc_factual",
        question=question,
        expected_answer=expected_answer,
        gold_sources=[GoldLocation(ticker="ROG", year=2024, pages=[2])],
        expected_behavior=expected_behavior,
        gold_match="all",
        notes=None,
    )


def make_hit(text: str = "Research and development 13,042") -> RetrievedHit:
    return RetrievedHit(
        ticker="ROG",
        company_name="Roche",
        sector="Pharma",
        exchange="SIX",
        year=2024,
        page_number=2,
        text=text,
        distance=0.1,
        chunk_id="c1",
        rerank_score=0.9,
    )


def make_tool_response(verdict_input: dict) -> MagicMock:
    """Fake anthropic response with a single record_verdict tool_use block."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = "record_verdict"
    block.input = verdict_input
    response = MagicMock()
    response.content = [block]
    return response


def make_judge(verdict_input: dict) -> tuple[Judge, MagicMock]:
    client = MagicMock()
    client.messages.create.return_value = make_tool_response(verdict_input)
    return Judge(anthropic_client=client, model="judge-test-model"), client


def test_judge_parses_verdict_fields():
    judge, _ = make_judge(
        {
            "behavior": "answered",
            "fabricated": False,
            "answer_correct": True,
            "reasoning": "matches reference",
        }
    )

    verdict = judge.judge(make_question(), "Roche R&D was CHF 13.0bn.", [make_hit()])

    assert verdict.question_id == "A001"
    assert verdict.behavior == "answered"
    assert verdict.answer_correct is True
    assert verdict.fabricated is False
    assert verdict.reasoning == "matches reference"


def test_judge_forces_record_verdict_tool_and_uses_configured_model():
    judge, client = make_judge(
        {"behavior": "answered", "fabricated": False, "answer_correct": True}
    )

    judge.judge(make_question(), "answer", [make_hit()])

    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "judge-test-model"
    assert kwargs["tool_choice"] == {"type": "tool", "name": "record_verdict"}
    assert kwargs["tools"][0]["name"] == "record_verdict"


def test_judge_refused_forces_answer_correct_null():
    # even if the model returns answer_correct, a refusal must null it out.
    judge, _ = make_judge(
        {"behavior": "refused", "fabricated": False, "answer_correct": True}
    )

    verdict = judge.judge(
        make_question(expected_behavior="refuse"), "I don't have that.", []
    )

    assert verdict.behavior == "refused"
    assert verdict.answer_correct is None


def test_judge_fabricated_passes_through():
    judge, _ = make_judge(
        {"behavior": "answered", "fabricated": True, "reasoning": "made up a number"}
    )

    verdict = judge.judge(make_question(), "Roche R&D was CHF 99bn.", [make_hit()])

    assert verdict.fabricated is True


def test_judge_user_message_includes_question_reference_and_excerpts():
    judge, client = make_judge(
        {"behavior": "answered", "fabricated": False, "answer_correct": True}
    )
    q = make_question(question="How much R&D?", expected_answer="CHF 13.0 billion")

    judge.judge(q, "the system answer", [make_hit(text="R&D 13,042")])

    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "How much R&D?" in content
    assert "CHF 13.0 billion" in content
    assert "the system answer" in content
    assert "[ROG 2024, p. 2]" in content
    assert "R&D 13,042" in content


def test_judge_formats_no_excerpts_placeholder():
    judge, client = make_judge(
        {"behavior": "refused", "fabricated": False}
    )

    judge.judge(make_question(), "no info", [])

    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "(no excerpts were provided to the system)" in content


def test_judge_raises_when_no_tool_call_in_response():
    client = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    response = MagicMock()
    response.content = [text_block]
    client.messages.create.return_value = response
    judge = Judge(anthropic_client=client, model="judge-test-model")

    with pytest.raises(ValueError, match="record_verdict"):
        judge.judge(make_question(), "answer", [make_hit()])


def test_judge_answer_delegates_to_default_judge(monkeypatch):
    fake = MagicMock(spec=Judge)
    sentinel = object()
    fake.judge.return_value = sentinel
    monkeypatch.setattr(judge_module, "_default_judge", fake)

    q = make_question()
    result = judge_answer(q, "answer", [make_hit()])

    assert result is sentinel
    fake.judge.assert_called_once_with(q, "answer", [make_hit()])


def test_default_judge_model_is_stronger_than_generator():
    # sanity: the judge should not default to the same model as the system under test.
    assert settings.judge_model != settings.llm_model
