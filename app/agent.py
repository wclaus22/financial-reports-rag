"""hand rolled agent to take apart more complex queries and solve more involved questions"""

import anthropic

from app.config import settings
from app.generation import SYSTEM_PROMPT
from app.models import RetrievedHit
from app.pipeline import QueryResult
from app.retrieval import Retriever

SEARCH_TOOL = {
    "name": "search",
    "description": "use this tool to retrieve relevant information from the financial reports database.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "the user's original question"},
            # tickers and years must be updated according to updates to the corpus
            "ticker": {
                "type": "string",
                "enum": ["ROG", "NOVN", "UBSG", "NESN", "ZURN"],
            },
            "year": {"type": "integer", "enum": [2021, 2022, 2023, 2024, 2025]},
        },
        "required": ["query"],
    },
}

AGENT_PROMPT = (
    SYSTEM_PROMPT
    + """

Corpus Manifest:
Company Tickers = ROG, NOVN, UBSG, NESN, ZURN
Years = 2021, 2022, 2023, 2024, 2025

Perform one search call per (company, year) tuple. Issue them in parallel for comparisons and trends.

State a figure ONLY if an excerpt for that exact (company, year) tuple is found. Say so explicitly if no excerpt is found for that tuple, rather than guessing or extrapolating.
NEVER infer or interpolate figures across (company, year) tuples.

"""
)


def _render_hits(hits: list[RetrievedHit]) -> str:
    """Format retrieved hits into a string for the agent to read."""
    if not hits:
        return "No relevant excerpts found for this search."

    blocks = []
    for hit in hits:
        header = f"[{hit.ticker} {hit.year}, p. {hit.page_number}]"
        blocks.append(f"{header}\n{hit.text}")

    context = "\n\n---\n\n".join(blocks)
    return context


class Agent:
    """Agent class to handle complex queries with multiple retrieval and generation steps."""

    def __init__(self, retriever: Retriever, anthropic_client=None) -> None:
        self.retriever = retriever
        self.client = anthropic_client or anthropic.Anthropic(
            api_key=settings.anthropic_api_key
        )

    def run(self, question: str, top_k: int = 5) -> QueryResult:
        """Run the agent on an input question, returning the final answer and the retrieved hits."""
        messages = [{"role": "user", "content": question}]
        union = []
        seen = set()

        for i in range(settings.agent_max_iterations):
            response = self.client.messages.create(
                model=settings.agent_model,
                max_tokens=1024,
                system=AGENT_PROMPT,
                tools=[SEARCH_TOOL],
                tool_choice={"type": "none"}
                # last iteration must be generation, even if the agent wants to search again
                if i == settings.agent_max_iterations - 1
                else {"type": "auto"},
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                break
            tool_results = []
            tool_calls = [
                block for block in response.content if block.type == "tool_use"
            ]
            for call in tool_calls:
                args = call.input
                hits = self.retriever.search(
                    query=args["query"],
                    ticker=args.get("ticker"),
                    year=args.get("year"),
                    top_k=top_k,
                )
                for h in hits:
                    if h.chunk_id not in seen:
                        seen.add(h.chunk_id)
                        union.append(h)
                tool_results.append(
                    {
                        "content": _render_hits(hits),
                        "type": "tool_result",
                        "tool_use_id": call.id,
                    }
                )
            messages.append({"role": "user", "content": tool_results})
        answer = "".join(
            block.text for block in response.content if block.type == "text"
        )
        return QueryResult(answer=answer, retrieved=union, grounding=union)
