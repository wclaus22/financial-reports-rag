"""metrics for the evaluation harness.

Two independent axes are scored per question:

1. Retrieval -- did the retrieved chunks cover the gold sources? Fully
   deterministic; computed here. Only applies to questions with gold sources
   (i.e. expected_behavior == "answer"). Refusal questions have no gold and are
   excluded from retrieval aggregates.

2. Behavior -- did the system do the right thing (answer correctly / refuse /
   either, and crucially NOT fabricate)? This needs an LLM judge, which lives in
   judge.py. This module does not call the judge; it consumes the verdict the
   judge produces (JudgeVerdict) and turns it into a pass/fail + aggregates.

Matching rules (see EvalQuestion.gold_match):
  - within a single gold source, hitting ANY of its listed pages satisfies it.
  - across gold sources, "all" requires every source hit, "any" requires one.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence, runtime_checkable

from evals.dataset import EvalQuestion, GoldLocation


@runtime_checkable
class RetrievedLike(Protocol):
    """structural type for a retrieved chunk (RetrievedHit or Source)."""

    ticker: str
    year: int
    page_number: int


@dataclass
class RetrievalScore:
    """retrieval quality for a single question."""

    question_id: str
    applicable: bool  # False for refusal questions (no gold sources to hit)
    source_hits: list[bool] = field(default_factory=list)
    source_recall: float = 0.0  # fraction of gold sources hit (diagnostic)
    passed: bool = False  # all()/any() over source_hits per gold_match
    first_hit_rank: int | None = None  # 1-based rank of first gold-matching chunk
    reciprocal_rank: float = 0.0  # 1/first_hit_rank, else 0


@dataclass
class Aggregate:
    """rolled-up scores over a group of questions."""

    n: int  # total questions in the group
    n_retrieval: int  # questions with gold sources (retrieval-applicable)
    retrieval_pass_rate: float  # fraction of applicable questions where passed
    mean_source_recall: float  # mean fractional source recall (applicable only)
    mean_reciprocal_rank: float  # MRR over applicable questions
    behavior_pass_rate: float  # fraction of all questions passing behavior


@dataclass
class JudgeVerdict:
    """what the LLM judge (judge.py) reports about the system's answer.

    `fabricated` is the load-bearing signal for the adversarial / sparse cases:
    asserting anything not supported by retrieved context is a hard failure,
    regardless of whether the system answered or refused.
    """

    question_id: str
    behavior: Literal["answered", "refused"]  # what the system actually did
    answer_correct: bool | None = None  # judged vs expected_answer; None if refused
    fabricated: bool = False  # asserted unsupported / hallucinated claims
    reasoning: str | None = None  # brief judge justification, for triage


def _pages_by_doc(
    retrieved: Sequence[RetrievedLike],
) -> dict[tuple[str, int], set[int]]:
    """collapse retrieved chunks into the set of pages hit per (ticker, year)."""
    out: dict[tuple[str, int], set[int]] = defaultdict(set)
    for hit in retrieved:
        out[(hit.ticker, hit.year)].add(hit.page_number)
    return out


def _source_hit(
    gold: GoldLocation, pages_by_doc: dict[tuple[str, int], set[int]]
) -> bool:
    """a gold source is hit if any of its listed pages was retrieved."""
    return bool(set(gold.pages) & pages_by_doc.get((gold.ticker, gold.year), set()))


def score_retrieval(
    q: EvalQuestion, retrieved: Sequence[RetrievedLike]
) -> RetrievalScore:
    """score how well `retrieved` (in rank order) covers q's gold sources."""
    if not q.gold_sources:
        # refusal / no-gold question: retrieval recall is undefined, skip it.
        return RetrievalScore(question_id=q.id, applicable=False)

    pages_by_doc = _pages_by_doc(retrieved)
    hits = [_source_hit(g, pages_by_doc) for g in q.gold_sources]
    recall = sum(hits) / len(hits)
    passed = all(hits) if q.gold_match == "all" else any(hits)

    # rank of the first retrieved chunk that lands on a gold page of a gold doc.
    wanted = {(g.ticker, g.year): set(g.pages) for g in q.gold_sources}
    first_rank: int | None = None
    for rank, hit in enumerate(retrieved, start=1):
        pages = wanted.get((hit.ticker, hit.year))
        if pages and hit.page_number in pages:
            first_rank = rank
            break

    return RetrievalScore(
        question_id=q.id,
        applicable=True,
        source_hits=hits,
        source_recall=recall,
        passed=passed,
        first_hit_rank=first_rank,
        reciprocal_rank=(1.0 / first_rank) if first_rank else 0.0,
    )


def behavior_passed(q: EvalQuestion, v: JudgeVerdict) -> bool:
    """did the system's behavior satisfy this question's expectation?"""
    if v.fabricated:
        return False  # fabrication never passes, in any mode

    match q.expected_behavior:
        case "refuse":
            return v.behavior == "refused"
        case "answer":
            return v.behavior == "answered" and bool(v.answer_correct)
        case "answer_or_refuse":
            return True
        case _:  # pragma: no cover - guarded by the Literal type
            raise ValueError(f"unknown expected_behavior: {q.expected_behavior}")


def _safe_mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def aggregate(
    questions: Sequence[EvalQuestion],
    retrieval: dict[str, RetrievalScore],
    behavior: dict[str, JudgeVerdict],
) -> Aggregate:
    """aggregate per-question scores; both dicts keyed by question id."""
    applicable = [retrieval[q.id] for q in questions if retrieval[q.id].applicable]
    behavior_passes = [
        behavior_passed(q, behavior[q.id]) for q in questions if q.id in behavior
    ]

    return Aggregate(
        n=len(questions),
        n_retrieval=len(applicable),
        retrieval_pass_rate=_safe_mean([float(r.passed) for r in applicable]),
        mean_source_recall=_safe_mean([r.source_recall for r in applicable]),
        mean_reciprocal_rank=_safe_mean([r.reciprocal_rank for r in applicable]),
        behavior_pass_rate=_safe_mean([float(p) for p in behavior_passes]),
    )


def aggregate_by_category(
    questions: Sequence[EvalQuestion],
    retrieval: dict[str, RetrievalScore],
    behavior: dict[str, JudgeVerdict],
) -> dict[str, Aggregate]:
    """one Aggregate per category, plus an "overall" entry."""
    by_cat: dict[str, list[EvalQuestion]] = defaultdict(list)
    for q in questions:
        by_cat[q.category].append(q)

    out = {cat: aggregate(qs, retrieval, behavior) for cat, qs in by_cat.items()}
    out["overall"] = aggregate(questions, retrieval, behavior)
    return out
