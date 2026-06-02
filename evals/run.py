"""Run the evaluation set end-to-end, in-process.

Drives the retriever + generator directly (no HTTP server required), so the
eval exercises the same components the /query endpoint does without the network
hop. For each question it:

  1. retrieves with Retriever.retrieve(question, top_k)
  2. scores retrieval deterministically against the gold sources (metrics)
  3. generates an answer with Generator.generate_answer(question, hits)
  4. optionally judges behaviour via the LLM judge in evals.judge

Retrieval is scored on the retriever's own slate (step 1) -- that is the truest
"did we fetch the right pages" signal, independent of generation/safety. The
judge instead sees the answer and the grounding the generator actually used.

Results are written to evals/results/eval_<utc-timestamp>.json and a per-category
summary is printed.

Usage:
    python -m evals.run                      # full set, with judge
    python -m evals.run --no-judge           # retrieval only (no LLM judge)
    python -m evals.run --category single_doc_factual --limit 3
    python -m evals.run --id A001 --id B002
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.config import settings
from app.generation import Generator
from app.models import RetrievedHit
from app.pipeline import run_query
from app.retrieval import Retriever
from app.safety import GenerationSafety, NoSafetyMechanism
from evals.dataset import EvalQuestion, load_eval_set
from evals.metrics import (
    JudgeVerdict,
    RetrievalScore,
    aggregate_by_category,
    behavior_passed,
    score_retrieval,
)

logger = logging.getLogger(__name__)

# A judge takes the question, the generated answer, and the grounding excerpts
# the generator actually used, and returns a JudgeVerdict. evals.judge supplies
# the implementation; run.py only depends on this shape.
JudgeFn = Callable[[EvalQuestion, str, list[RetrievedHit]], JudgeVerdict]


@dataclass
class QuestionResult:
    """everything we record for one evaluated question."""

    id: str
    category: str
    question: str
    expected_answer: str
    expected_behavior: str
    gold_match: str
    answer: str
    retrieval: RetrievalScore
    verdict: JudgeVerdict | None = None
    behavior_pass: bool | None = None
    error: str | None = None
    # compact view of what was retrieved, for eyeballing failures
    retrieved: list[dict] = field(default_factory=list)


def _compact_hit(h: RetrievedHit) -> dict:
    return {
        "ticker": h.ticker,
        "year": h.year,
        "page": h.page_number,
        "rerank_score": h.rerank_score,
    }


def evaluate_question(
    q: EvalQuestion,
    retriever: Retriever,
    generator: Generator,
    top_k: int,
    judge: JudgeFn | None,
) -> QuestionResult:
    """retrieve -> score -> generate -> (optionally) judge, for one question."""
    query = run_query(q.question, top_k, retriever, generator)
    # score retrieval on the raw retriever slate, NOT the post-safety grounding.
    retrieval = score_retrieval(q, query.retrieved)
    answer, grounding = query.answer, query.grounding

    result = QuestionResult(
        id=q.id,
        category=q.category,
        question=q.question,
        expected_answer=q.expected_answer,
        expected_behavior=q.expected_behavior,
        gold_match=q.gold_match,
        answer=answer,
        retrieval=retrieval,
        retrieved=[_compact_hit(h) for h in query.retrieved],
    )

    if judge is not None:
        verdict = judge(q, answer, grounding)
        result.verdict = verdict
        result.behavior_pass = behavior_passed(q, verdict)

    return result


def load_judge() -> JudgeFn | None:
    """import the LLM judge if it's implemented; else None (retrieval-only)."""
    try:
        from evals.judge import judge_answer
    except (ImportError, AttributeError):
        logger.warning(
            "evals.judge.judge_answer not available -- running retrieval only. "
            "Implement it or pass --no-judge to silence this."
        )
        return None
    return judge_answer


def select_questions(
    questions: list[EvalQuestion],
    categories: list[str] | None,
    ids: list[str] | None,
    limit: int | None,
) -> list[EvalQuestion]:
    """apply --category / --id / --limit filters."""
    selected = questions
    if categories:
        wanted = set(categories)
        selected = [q for q in selected if q.category in wanted]
    if ids:
        wanted_ids = set(ids)
        selected = [q for q in selected if q.id in wanted_ids]
    if limit is not None:
        selected = selected[:limit]
    return selected


def _fmt_pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def print_report(by_cat: dict, judged: bool) -> None:
    """print a per-category summary table, overall last."""
    header = (
        f"{'category':<26} {'n':>3} {'ret_n':>5} "
        f"{'ret_pass':>9} {'recall':>7} {'mrr':>6} {'behav':>7}"
    )
    print("\n" + header)
    print("-" * len(header))

    # keep "overall" at the bottom
    cats = [c for c in by_cat if c != "overall"] + ["overall"]
    for cat in cats:
        a = by_cat[cat]
        behav = _fmt_pct(a.behavior_pass_rate) if judged else "   n/a"
        if cat == "overall":
            print("=" * len(header))
        print(
            f"{cat:<26} {a.n:>3} {a.n_retrieval:>5} "
            f"{_fmt_pct(a.retrieval_pass_rate):>9} "
            f"{_fmt_pct(a.mean_source_recall):>7} "
            f"{a.mean_reciprocal_rank:>6.3f} {behav:>7}"
        )


def print_failures(results: list[QuestionResult], judged: bool) -> None:
    """list questions that failed retrieval or behaviour, for quick triage."""
    fails = []
    for r in results:
        if r.error:
            fails.append((r.id, f"ERROR: {r.error}"))
        elif r.retrieval.applicable and not r.retrieval.passed:
            fails.append((r.id, f"retrieval miss (recall {r.retrieval.source_recall:.2f})"))
        elif judged and r.behavior_pass is False:
            v = r.verdict
            why = "fabricated" if v and v.fabricated else f"behaved={v.behavior if v else '?'}"
            fails.append((r.id, f"behaviour fail ({why})"))
    if fails:
        print(f"\nFailures ({len(fails)}):")
        for qid, why in fails:
            print(f"  {qid}: {why}")


def write_results(
    results: list[QuestionResult], by_cat: dict, out_path: Path, meta: dict
) -> None:
    payload = {
        "meta": meta,
        "summary": {cat: asdict(a) for cat, a in by_cat.items()},
        "results": [asdict(r) for r in results],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {len(results)} results to {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the financial-reports RAG eval set.")
    p.add_argument("--eval-set", default="evals/eval_set.json", help="path to eval_set.json")
    p.add_argument("--top-k", type=int, default=settings.top_k, help="retriever top_k")
    p.add_argument("--category", action="append", help="filter by category (repeatable)")
    p.add_argument("--id", action="append", dest="ids", help="filter by question id (repeatable)")
    p.add_argument("--limit", type=int, default=None, help="cap number of questions")
    p.add_argument("--no-judge", action="store_true", help="skip the LLM behaviour judge")
    p.add_argument("--out", default=None, help="results JSON path (default: timestamped)")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    questions = load_eval_set(args.eval_set)
    questions = select_questions(questions, args.category, args.ids, args.limit)
    if not questions:
        raise SystemExit("No questions selected after filtering.")

    judge = None if args.no_judge else load_judge()
    judged = judge is not None

    retriever = Retriever()
    generator = Generator(GenerationSafety(NoSafetyMechanism()))

    results: list[QuestionResult] = []
    for i, q in enumerate(questions, start=1):
        logger.info("[%d/%d] %s %s", i, len(questions), q.id, q.question[:70])
        try:
            results.append(evaluate_question(q, retriever, generator, args.top_k, judge))
        except Exception as e:  # one bad question shouldn't sink the whole run
            logger.exception("Question %s failed", q.id)
            results.append(
                QuestionResult(
                    id=q.id,
                    category=q.category,
                    question=q.question,
                    expected_answer=q.expected_answer,
                    expected_behavior=q.expected_behavior,
                    gold_match=q.gold_match,
                    answer="",
                    retrieval=RetrievalScore(question_id=q.id, applicable=False),
                    error=str(e),
                )
            )

    retrieval_by_id = {r.id: r.retrieval for r in results}
    verdict_by_id = {r.id: r.verdict for r in results if r.verdict is not None}
    by_cat = aggregate_by_category(questions, retrieval_by_id, verdict_by_id)

    print_report(by_cat, judged)
    print_failures(results, judged)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.out) if args.out else Path(f"evals/results/eval_{timestamp}.json")
    meta = {
        "timestamp_utc": timestamp,
        "eval_set": args.eval_set,
        "top_k": args.top_k,
        "judged": judged,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "rerank_model": settings.rerank_model,
        "n_questions": len(questions),
    }
    write_results(results, by_cat, out_path, meta)


if __name__ == "__main__":
    main()
