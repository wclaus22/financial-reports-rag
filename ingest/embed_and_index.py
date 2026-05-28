"""
End-to-end process of ingestion:
1. Parse PDFs into page-level text with metadata
2. Split page-level text into retrieval sized chunks while preserving metadata
3. Embed chunks and index in a vector database
"""

import time

import chromadb
import tqdm
import voyageai

from app.config import settings
from ingest.chunk import chunk_pages
from ingest.embed import embed_batch
from ingest.models import Chunk
from ingest.parse import parse_pdf

EMBEDDING_BATCH_SIZE = 64
RATE_LIMIT_SLEEP_SEC = 0.1


def index_chunks(
    chunks: list[Chunk],
    voyage_client: voyageai.Client,
    collection: "chromadb.Collection",
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> None:
    """Embed `chunks` in batches and add them to the chroma collection."""
    for i in tqdm.tqdm(
        range(0, len(chunks), batch_size),
        desc="Embedding and indexing chunks",
        total=(len(chunks) + batch_size - 1) // batch_size,
    ):
        batch = chunks[i : i + batch_size]
        texts = [chunk.text for chunk in batch]
        embeddings = embed_batch(voyage_client, texts)
        collection.add(
            ids=[chunk.chunk_id for chunk in batch],
            metadatas=[
                {
                    "ticker": chunk.ticker,
                    "company_name": chunk.company_name,
                    "sector": chunk.sector,
                    "exchange": chunk.exchange,
                    "year": chunk.year,
                    "page_number": chunk.page_number,
                }
                for chunk in batch
            ],
            embeddings=embeddings,
            documents=texts,
        )
        time.sleep(RATE_LIMIT_SLEEP_SEC)


def main() -> None:
    """main function to run the parsing and ingestion process"""

    print("STEP 1/3: Parsing PDFs")
    pages = parse_pdf(settings.metadata_path)

    print("=" * 60)
    print("STEP 2/3: Chunking pages")
    chunks = chunk_pages(pages)

    print("=" * 60)
    print("STEP 3/3: Embedding (Voyage) + indexing (Chroma)")
    print("=" * 60)
    print(f"Embedding Model: {settings.embedding_model}")
    print(f"Chroma Persist Directory: {settings.chroma_persist_directory}")

    voyage = voyageai.Client(api_key=settings.voyage_api_key)
    chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_directory)

    # reset collection if it already exists
    try:
        chroma_client.delete_collection(settings.collection_name)
    except chromadb.errors.NotFoundError:
        pass

    collection = chroma_client.create_collection(
        name=settings.collection_name, metadata={"hnsw:space": "cosine"}
    )

    index_chunks(chunks, voyage, collection)

    print(f"Ingestion complete! Collection size: {collection.count()} chunks.")


if __name__ == "__main__":
    main()
