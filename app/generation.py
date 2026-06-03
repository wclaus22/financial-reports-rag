"""
generation module for generating responses based on retrieved documents.
"""

from typing import Tuple

import anthropic

from app.config import settings
from app.models import RetrievedHit
from app.safety import GenerationSafety

SYSTEM_PROMPT = """You are a financial analyst assistant answering questions about Swiss-listed company annual reports.

You are given a question and a set of retrieved excerpts from the reports. Answer the question using ONLY the information in the excerpts. If the excerpts do not contain enough information to answer, say so explicitly rather than guessing.

When citing information, reference the source as [Ticker Year, p. PAGE]. For example: "Roche's R&D spending grew 7% in 2024 [ROG 2024, p. 23]."

Be precise and concise. Quote specific figures where the excerpts provide them. If excerpts conflict, surface the conflict rather than picking one.

Grounding discipline: state a figure ONLY if an excerpt for that exact company AND year is present. If a requested company or year is absent from the excerpts, say so explicitly rather than guessing. NEVER infer, interpolate, or carry a figure across a different year or company. Always cite the source as [Ticker Year, p. PAGE]."""


class Generator:
    """Generator class for generating responses based on retrieved documents."""

    def __init__(
        self,
        safety: GenerationSafety,
        anthropic_client: "anthropic.Anthropic | None" = None,
    ) -> None:
        self.client = anthropic_client or anthropic.Anthropic(
            api_key=settings.anthropic_api_key
        )
        self.safety = safety

    def generate_answer(
        self, query: str, hits: list[RetrievedHit]
    ) -> Tuple[str, list[RetrievedHit]]:
        """Generate an answer to the user query based on the retrieved hits."""

        filtered_hits = []
        for hit in hits:
            filtered_hits = self.safety.add_hit(hit, filtered_hits)
        filtered_hits = self.safety.check_hits(filtered_hits)

        blocks = []
        for hit in filtered_hits:
            header = f"[{hit.ticker} {hit.year}, p. {hit.page_number}]"
            blocks.append(f"{header}\n{hit.text}")

        context = "\n\n---\n\n".join(blocks)

        user_promt = (
            f"Question: {query}\n\n"
            f"Retrieved excerpts:\n\n{context}\n\n"
            f"Answer the question using only the excerpts above."
        )

        response = self.client.messages.create(
            model=settings.llm_model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_promt}],
        )

        # Concatenate all text blocks in the response
        return "".join(
            block.text for block in response.content if block.type == "text"
        ), filtered_hits
