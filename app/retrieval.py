"""
retrieval module for retrieving relevant documents based on user queries.
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
        Retrieve relevant documents based on user query.

        Args:
            query (str): User query.
            top_k (int | None): Number of top results to return. Defaults to settings.top_k.

        Returns:
            list[dict]: List of relevant documents with metadata and distance.
        """
        top_k = top_k or settings.top_k

        query_emb = self.voyage.embed(
            [query], model=settings.embedding_model, input_type="query"
        ).embeddings[0]

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
        )

        hits: list[dict] = []
        for chunk_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append(
                RetrievedHit(
                    ticker=meta["ticker"],
                    company_name=meta["company_name"],
                    sector=meta["sector"],
                    exchange=meta["exchange"],
                    year=meta["year"],
                    page_number=meta["page_number"],
                    text=doc,
                    distance=dist,
                    chunk_id=chunk_id,
                )
            )
        return hits
