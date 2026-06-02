"""LLM judge for the evaluation harness.

Given a question, the system's answer, and the grounding excerpts the generator
actually used, the judge reports three INDEPENDENT signals as a JudgeVerdict:

  - behavior:        did the system answer or refuse?
  - fabricated:      did the answer assert claims unsupported by the excerpts?
  - answer_correct:  (only if answered) is the answer consistent with the
                     reference answer?

The judge is deliberately NOT told the question's expected_behavior, so it
grades what actually happened rather than what was hoped for. metrics.behavior_passed
combines this verdict with the expected_behavior to decide pass/fail.

Fabrication is judged ONLY against the excerpts, never the reference answer:
a correct-sounding figure that is not in the grounding is still a hallucination
for a RAG system.
"""

import logging

import anthropic

from app.config import settings
from app.models import RetrievedHit
from evals.dataset import EvalQuestion
from evals.metrics import JudgeVerdict

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a strict, impartial evaluator (judge) for a financial-reports RAG system.

You are given a QUESTION, a REFERENCE answer (what a correct response should contain, or the behaviour expected for adversarial/unanswerable questions), the system's ANSWER, and the EXCERPTS the system was given as grounding.

Assess three things INDEPENDENTLY and record them with the `record_verdict` tool:

1. behavior — "answered" if the system gave a substantive answer to the question; "refused" if it declined, said it lacks the information, said the corpus does not cover the topic, flagged a false premise, or otherwise did not provide a substantive factual answer.

2. fabricated — true if the ANSWER asserts any specific factual claim (a figure, name, date, or event) that is NOT supported by the EXCERPTS. Judge this ONLY against the EXCERPTS, never the reference answer or outside knowledge. A refusal, or an explicit "I don't have that information", is never fabrication. Statements that are clearly hedged and grounded in the excerpts are not fabrication. Be strict: an unsupported number or named entity counts as fabricated.

3. answer_correct — ONLY when behavior is "answered". true if the ANSWER is factually consistent with the REFERENCE answer (judge semantically: minor wording, rounding, or unit-formatting differences are fine; the key figure/fact must match). false if it contradicts or omits the key fact. If the system refused, leave answer_correct null.

Always provide brief reasoning. Do not use outside knowledge to fill gaps in the excerpts."""

JUDGE_TOOL = {
    "name": "record_verdict",
    "description": "Record the structured verdict for the system's answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "behavior": {
                "type": "string",
                "enum": ["answered", "refused"],
                "description": "Whether the system gave a substantive answer or refused/declined.",
            },
            "fabricated": {
                "type": "boolean",
                "description": "True if the answer asserts a specific claim not supported by the excerpts.",
            },
            "answer_correct": {
                "type": "boolean",
                "description": "Only when answered: is the answer consistent with the reference answer? Omit if refused.",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief justification for the verdict (one or two sentences).",
            },
        },
        "required": ["behavior", "fabricated", "reasoning"],
    },
}


def _format_excerpts(grounding: list[RetrievedHit]) -> str:
    """Render grounding hits the way the generator saw them, for the judge."""
    if not grounding:
        return "(no excerpts were provided to the system)"
    blocks = []
    for hit in grounding:
        header = f"[{hit.ticker} {hit.year}, p. {hit.page_number}]"
        blocks.append(f"{header}\n{hit.text}")
    return "\n\n---\n\n".join(blocks)


def _build_user_message(
    question: EvalQuestion, answer: str, grounding: list[RetrievedHit]
) -> str:
    return (
        f"QUESTION:\n{question.question}\n\n"
        f"REFERENCE answer:\n{question.expected_answer}\n\n"
        f"System ANSWER:\n{answer}\n\n"
        f"EXCERPTS the system was given:\n\n{_format_excerpts(grounding)}\n\n"
        "Evaluate the system ANSWER and call record_verdict."
    )


class Judge:
    """LLM judge. Mirrors Generator's injectable-client shape for testability."""

    def __init__(
        self,
        anthropic_client: "anthropic.Anthropic | None" = None,
        model: str | None = None,
    ) -> None:
        self.client = anthropic_client or anthropic.Anthropic(
            api_key=settings.anthropic_api_key
        )
        self.model = model or settings.judge_model

    def judge(
        self, question: EvalQuestion, answer: str, grounding: list[RetrievedHit]
    ) -> JudgeVerdict:
        """Return a JudgeVerdict for one (question, answer, grounding) triple."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[JUDGE_TOOL],
            tool_choice={"type": "tool", "name": "record_verdict"},
            messages=[
                {
                    "role": "user",
                    "content": _build_user_message(question, answer, grounding),
                }
            ],
        )

        verdict_input = _extract_tool_input(response)

        behavior = verdict_input["behavior"]
        # correctness is only meaningful for an actual answer; force null on refusal.
        answer_correct = (
            verdict_input.get("answer_correct") if behavior == "answered" else None
        )

        return JudgeVerdict(
            question_id=question.id,
            behavior=behavior,
            answer_correct=answer_correct,
            fabricated=verdict_input["fabricated"],
            reasoning=verdict_input.get("reasoning"),
        )


def _extract_tool_input(response: "anthropic.types.Message") -> dict:
    """Pull the record_verdict tool input out of a forced-tool response."""
    for block in response.content:
        if block.type == "tool_use" and block.name == "record_verdict":
            return block.input
    raise ValueError("Judge response did not contain a record_verdict tool call.")


# Module-level convenience matching the JudgeFn contract in evals.run. A single
# lazily-built Judge is reused across questions so we don't reconstruct the
# Anthropic client per call (mirrors main.py's singleton pattern).
_default_judge: Judge | None = None


def judge_answer(
    question: EvalQuestion, answer: str, grounding: list[RetrievedHit]
) -> JudgeVerdict:
    """Judge one answer using a shared default Judge instance."""
    global _default_judge
    if _default_judge is None:
        _default_judge = Judge()
    return _default_judge.judge(question, answer, grounding)
