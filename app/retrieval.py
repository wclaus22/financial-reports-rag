"""
retrieval module for retrieving relevant documents based on user queries.

Two-stage retrieval: first fetch a candidate pool (settings.rerank_candidates,
default 50) by vector similarity in Chroma, then rerank with Voyage's
cross-encoder (settings.rerank_model) and return the top-K reordered hits.
"""

import chromadb
import voyageai

from app.config import settings
from app.models import RetrievedHit


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

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedHit]:
        """
        Retrieve relevant documents for a user query.

        Stage 1: vector search in Chroma fetches `settings.rerank_candidates`
                 candidates by cosine similarity.
        Stage 2: Voyage rerank scores each candidate against the query with a
                 cross-encoder and returns the top `top_k` by relevance.

        Args:
            query: User query.
            top_k: Number of top results to return after reranking.
                   Defaults to settings.top_k.

        Returns:
            list[RetrievedHit] ordered by rerank relevance (best first), each
            carrying both the original vector distance and the rerank score.
        """
        top_k = top_k or settings.top_k

        query_emb = self.voyage.embed(
            [query], model=settings.embedding_model, input_type="query"
        ).embeddings[0]

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=settings.rerank_candidates,
        )

        docs = results["documents"][0]
        if not docs:
            return []

        rerank = self.voyage.rerank(
            query=query,
            documents=docs,
            model=settings.rerank_model,
            top_k=top_k,
        )

        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        hits: list[RetrievedHit] = []
        for r in rerank.results:
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
