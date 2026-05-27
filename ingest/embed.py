"""embedding logic for chunks of text"""

import voyageai

from app.config import settings


def embed_batch(voyage_client: voyageai.Client, texts: list[str]) -> list[list[float]]:
    """embed a batch of texts using the Voyage API with rate limiting"""
    result = voyage_client.embed(
        texts, model=settings.embedding_model, input_type="document"
    )
    return result.embeddings
