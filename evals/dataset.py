"""data models for evaluation"""

import json
from dataclasses import dataclass
from typing import Literal


@dataclass
class GoldLocation:
    """a gold source for an evaluation question"""

    ticker: str
    year: int
    pages: list[int]


@dataclass
class EvalQuestion:
    """an evaluation question"""

    id: str
    category: Literal[
        "single_doc_factual",
        "cross_doc_comparative",
        "multi_year_trend",
        "table_bound_numeric",
        "adversarial_keyword_trap",
        "adversarial_out_of_corpus",
        "adversarial_false_premise",
        "adversarial_injection",
        "adversarial_time_sensitive",
        "adversarial_sparse_topic",
    ]
    question: str
    expected_answer: str
    gold_sources: list[GoldLocation]
    expected_behavior: Literal["answer", "refuse", "answer_or_refuse"]
    # how the gold_sources combine for retrieval scoring:
    #   "all" -> every gold source must be hit (e.g. cross-doc / multi-year)
    #   "any" -> hitting one gold source is sufficient (e.g. same fact in two reports)
    # within a single source, hitting ANY of its listed pages satisfies that source.
    gold_match: Literal["all", "any"] = "all"
    notes: str | None = None


def load_eval_set(path: str = "evals/eval_set.json") -> list[EvalQuestion]:
    """load the evaluation set from a JSON file"""
    with open(path) as f:
        data = json.load(f)
    return [
        EvalQuestion(
            id=item["id"],
            category=item["category"],
            question=item["question"],
            expected_answer=item["expected_answer"],
            gold_sources=[GoldLocation(**gs) for gs in item.get("gold_sources", [])],
            expected_behavior=item["expected_behavior"],
            gold_match=item.get("gold_match", "all"),
            notes=item.get("notes"),
        )
        for item in data["questions"]
    ]
