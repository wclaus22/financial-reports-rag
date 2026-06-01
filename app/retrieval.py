"""
retrieval module for retrieving relevant documents based on user queries.

Three-stage retrieval:
  1. Year-aware candidate fetch from Chroma. Single-year / no-year queries
     pull one pool of `settings.rerank_candidates` candidates by vector
     similarity. Multi-year queries (e.g. "between 2021 and 2025",
     "2020-2024", "in 2021 and 2023") fan out into one per-year Chroma query
     with `where={"year": Y}` so the rerank pool covers each referenced year.
  2. Voyage rerank-2.5 scores each candidate against the query with a
     cross-encoder.
  3. Top-K reranked hits are returned.
"""

import re

import chromadb
import voyageai

from app.config import settings
from app.models import RetrievedHit

# Spelled-out year ranges. Each branch captures (low_year, high_year).
_YEAR_RANGE = re.compile(
    r"(?:between|from)\s+(\d{4})\s+(?:and|to)\s+(\d{4})"
    r"|(\d{4})\s*[-–]\s*(\d{4})"
    r"|(\d{4})\s+to\s+(\d{4})",
    re.IGNORECASE,
)

# A standalone 4-digit year token in the plausible report range.
_BARE_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")

# Cap to avoid absurd ranges (e.g. "from 1900 to 2025") fanning out into
# hundreds of Chroma queries.
_MAX_RANGE_SPAN = 20


def _extract_query_years(query: str) -> list[int]:
    """Return the list of years referenced by the query, if it's a *multi*-year
    intent — a range ("between 2021 and 2025") OR an explicit list of two or
    more years ("in 2021 and 2023"). Single-year and no-year queries return [].
    """
    m = _YEAR_RANGE.search(query)
    if m:
        groups = [g for g in m.groups() if g]
        if len(groups) >= 2:
            lo, hi = sorted([int(groups[0]), int(groups[1])])
            if hi - lo > _MAX_RANGE_SPAN:
                return []
            return list(range(lo, hi + 1))

    bare = sorted(set(int(y) for y in _BARE_YEAR.findall(query)))
    return bare if len(bare) >= 2 else []


class Retriever:
    """
    Retriever class for retrieving relevant documents based on user queries.
    """

    def __init__(
        self,
        voyage_client: "voyageai.Client | None" = None,
        collection: "chromadb.Collection | None" = None,
    ) -> None:
        self.voyage = voyage_client or voyageai.Client(api_key=settings.voyage_api_key)
        if collection is None:
            chroma_client = chromadb.PersistentClient(
                path=settings.chroma_persist_directory
            )
            collection = chroma_client.get_collection(settings.collection_name)
        self.collection = collection

    def _fetch_candidates(
        self, query_emb: list[float], years: list[int]
    ) -> tuple[list[str], list[dict], list[float], list[str]]:
        """Pull rerank candidates from Chroma.

        For multi-year queries, fans out into one filtered query per year and
        merges (deduped by chunk_id). For single-year / no-year queries, does
        one unfiltered query of `settings.rerank_candidates` size.
        """
        if not years:
            res = self.collection.query(
                query_embeddings=[query_emb],
                n_results=settings.rerank_candidates,
            )
            return (
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
                res["ids"][0],
            )

        per_year = settings.rerank_candidates_per_year
        docs: list[str] = []
        metas: list[dict] = []
        dists: list[float] = []
        ids: list[str] = []
        seen: set[str] = set()
        for y in years:
            res = self.collection.query(
                query_embeddings=[query_emb],
                n_results=per_year,
                where={"year": y},
            )
            for i, cid in enumerate(res["ids"][0]):
                if cid in seen:
                    continue
                seen.add(cid)
                docs.append(res["documents"][0][i])
                metas.append(res["metadatas"][0][i])
                dists.append(res["distances"][0][i])
                ids.append(cid)
        return docs, metas, dists, ids

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedHit]:
        """
        Retrieve relevant documents for a user query.

        Args:
            query: User query.
            top_k: Number of top results to return after reranking.
                   Defaults to settings.top_k.

        Returns:
            list[RetrievedHit] ordered by rerank relevance (best first), each
            carrying both the original vector distance and the rerank score.
            For multi-year queries a per-year diversity cap is applied so each
            referenced year is represented in the returned slate (the
            cross-encoder otherwise biases toward chunks whose year matches a
            year mentioned literally in the query).
        """
        top_k = top_k or settings.top_k

        query_emb = self.voyage.embed(
            [query], model=settings.embedding_model, input_type="query"
        ).embeddings[0]

        years = _extract_query_years(query)
        docs, metadatas, distances, ids = self._fetch_candidates(query_emb, years)

        if not docs:
            return []

        # For multi-year queries the cross-encoder still skews toward chunks
        # mentioning years that appear literally in the query, so a small
        # over-fetch is not enough to guarantee per-year coverage. Rerank the
        # full pool and let the per-year cap below pick a diverse slate.
        rerank_n = len(docs) if years else top_k
        rerank = self.voyage.rerank(
            query=query,
            documents=docs,
            model=settings.rerank_model,
            top_k=rerank_n,
        )

        if years:
            per_year_cap = max(1, -(-top_k // len(years)))  # ceil division
            year_counts: dict[int, int] = {}
            selected = []
            for r in rerank.results:
                y = metadatas[r.index]["year"]
                if year_counts.get(y, 0) >= per_year_cap:
                    continue
                year_counts[y] = year_counts.get(y, 0) + 1
                selected.append(r)
                if len(selected) >= top_k:
                    break
        else:
            selected = list(rerank.results)

        hits: list[RetrievedHit] = []
        for r in selected:
            i = r.index
            meta = metadatas[i]
            hits.append(
                RetrievedHit(
                    ticker=meta["ticker"],
                    company_name=meta["company_name"],
                    sector=meta["sector"],
                    exchange=meta["exchange"],
                    year=meta["year"],
                    page_number=meta["page_number"],
                    text=docs[i],
                    distance=distances[i],
                    chunk_id=ids[i],
                    rerank_score=r.relevance_score,
                )
            )
        return hits
